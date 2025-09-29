import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import logging
import re
import joblib
import os
import json
import argparse

logging.basicConfig(filename="../logs/training.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Train and evaluate GHG disclosure classifier.")
parser.add_argument('--validation', type=str, required=True, help="Path to the validation CSV file.")
args = parser.parse_args()


TRAIN_DATASET_PATH = "../data/dataset.csv"
try:
    df_train = pd.read_csv(TRAIN_DATASET_PATH)
    if 'label' not in df_train.columns or 'text_excerpt' not in df_train.columns:
        raise ValueError("Training dataset must contain 'label' and 'text_excerpt' columns.")
    logger.info(f"Loaded training dataset with {len(df_train)} entries.")
    class_dist = df_train['label'].value_counts().to_dict()
    logger.info(f"Training class distribution: {class_dist}")
except FileNotFoundError:
    print(f"Error: {TRAIN_DATASET_PATH} not found. Please run dataset_creator.py first.")
    logger.error(f"Training dataset file {TRAIN_DATASET_PATH} not found.")
    exit(1)
except ValueError as e:
    print(f"Error: {e}")
    logger.error(f"Invalid training dataset format: {e}")
    exit(1)

try:
    df_val = pd.read_csv(args.validation)
    if 'label' not in df_val.columns or 'text_excerpt' not in df_val.columns:
        raise ValueError("Validation dataset must contain 'label' and 'text_excerpt' columns.")
    logger.info(f"Loaded validation dataset with {len(df_val)} entries.")
    y_val = df_val['label'].values
except FileNotFoundError:
    print(f"Error: {args.validation} not found.")
    logger.error(f"Validation dataset file {args.validation} not found.")
    exit(1)
except ValueError as e:
    print(f"Error: {e}")
    logger.error(f"Invalid validation dataset format: {e}")
    exit(1)

y_train = df_train['label'].values


def preprocess_text(text):
    text = re.sub(r'\s+', ' ', str(text).lower().strip())
    text = re.sub(r'[^\w\s]', '', text)
    return text

df_train['text_excerpt'] = df_train['text_excerpt'].apply(preprocess_text)
df_val['text_excerpt'] = df_val['text_excerpt'].apply(preprocess_text)


tfidf_vectorizer = TfidfVectorizer(max_features=600, stop_words='english', ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['text_excerpt'])
X_val_tfidf = tfidf_vectorizer.transform(df_val['text_excerpt'])


def get_bert_embeddings(texts, batch_size=8):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

print("Generating BERT embeddings for training and validation data (this may take time)...")
X_train_bert = get_bert_embeddings(df_train['text_excerpt'].tolist())
X_val_bert = get_bert_embeddings(df_val['text_excerpt'].tolist())
logger.info(f"Generated BERT embeddings: Train shape {X_train_bert.shape}, Val shape {X_val_bert.shape}")

pca = PCA(n_components=15) 
X_train_bert_res_pca = pca.fit_transform(X_train_bert)
X_val_bert_res_pca = pca.transform(X_val_bert)


X_train_bert_res = X_train_bert_res_pca
smote = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=42)
rus = RandomUnderSampler(random_state=42)
X_train_tfidf_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)
X_train_bert_res, y_train_res = rus.fit_resample(X_train_bert_res, y_train_res)
logger.info(f"Post-resampling training size: {len(y_train_res)}, distribution: {pd.Series(y_train_res).value_counts().to_dict()}")


param_grid_lr = {'C': [0.1, 0.5, 1.0]}
param_grid_svm = {'C': [0.1, 0.5, 1.0]}

lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'), param_grid_lr, cv=5, scoring='f1_macro')
svm = GridSearchCV(SVC(kernel='linear', random_state=42, probability=True, class_weight='balanced'), param_grid_svm, cv=5, scoring='f1_macro')

models = {
    'Logistic Regression': lr,
    'SVM': svm,
    'DistilBERT': None  
}

results = {}
for name, model in models.items():
    print(f"\nTuning and Training {name}...")
    if name != 'DistilBERT':
        model.fit(X_train_tfidf_res, y_train_res)
        y_pred_val = model.predict(X_val_tfidf)
        X_train_used = X_train_tfidf_res

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model.best_estimator_, X_train_used, y_train_res, cv=cv, scoring='f1_macro')
        print(f"{name} - Best Params: {model.best_params_}")
        print(f"{name} - Cross-validation F1 Macro: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
        logger.info(f"{name} - Best Params: {model.best_params_}")
        logger.info(f"{name} - CV F1 Macro: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")

        # Validation
        print(f"\n{name} Final Validation Classification Report:")
        print(classification_report(y_val, y_pred_val, target_names=['Label 0', 'Label 1'], zero_division=0))
        print(f"Final Validation Confusion Matrix:\n{confusion_matrix(y_val, y_pred_val)}")

        logger.info(f"{name} Validation Report: {classification_report(y_val, y_pred_val, output_dict=True, zero_division=0)}")
        logger.info(f"{name} Validation Confusion Matrix: {confusion_matrix(y_val, y_pred_val)}")

        results[name] = {'y_val': y_val, 'y_pred_val': y_pred_val}


class DistilBERTClassifier(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=64, output_dim=2):  
        super(DistilBERTClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layer(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBERTClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)


X_train_bert_tensor = torch.FloatTensor(X_train_bert_res).to(device)
y_train_tensor = torch.LongTensor(y_train_res).to(device)
train_dataset = TensorDataset(X_train_bert_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


X_val_bert_tensor = torch.FloatTensor(X_val_bert_res_pca).to(device)
y_val_tensor = torch.LongTensor(y_val).to(device)
val_dataset = TensorDataset(X_val_bert_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=16)


model.train()
best_val_loss = float('inf')
patience = 3
trigger_times = 0
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()


    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            val_loss += criterion(outputs, batch_y).item()

    val_loss /= len(val_loader)
    logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), f"../models/best_model_DistilBERT.pth")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


model.load_state_dict(torch.load(f"../models/best_model_DistilBERT.pth"))
model.eval()
with torch.no_grad():
    outputs = model(X_val_bert_tensor)
    _, y_pred_val = torch.max(outputs, 1)
    y_pred_val = y_pred_val.cpu().numpy()

print("\nDistilBERT Final Validation Classification Report:")
print(classification_report(y_val, y_pred_val, target_names=['Label 0', 'Label 1'], zero_division=0))
print(f"Final Validation Confusion Matrix:\n{confusion_matrix(y_val, y_pred_val)}")
logger.info(f"DistilBERT Validation Report: {classification_report(y_val, y_pred_val, output_dict=True, zero_division=0)}")
logger.info(f"DistilBERT Validation Confusion Matrix: {confusion_matrix(y_val, y_pred_val)}")
results['DistilBERT'] = {'y_val': y_val, 'y_pred_val': y_pred_val}


best_model_name = max(
    results,
    key=lambda k: classification_report(results[k]['y_val'], results[k]['y_pred_val'], output_dict=True, zero_division=0)['macro avg']['f1-score']
)
if best_model_name != 'DistilBERT':
    best_model = models[best_model_name].best_estimator_
    joblib.dump(best_model, f"../models/best_model_{best_model_name.replace(' ', '_')}.pkl")
    joblib.dump(tfidf_vectorizer, '../models/tfidf_vectorizer.pkl')
else:
    logger.info(f"Saved best model: {best_model_name}")


best_preds = results[best_model_name]['y_pred_val']
accuracy = accuracy_score(y_val, best_preds)
precision = precision_score(y_val, best_preds, average="macro", zero_division=0)
recall = recall_score(y_val, best_preds, average="macro", zero_division=0)
f1 = f1_score(y_val, best_preds, average="macro", zero_division=0)


files_used = len(df_train) 
exchanges = df_train['exchange'].dropna().unique().tolist()

output = {
    "files_used_for_training": files_used,
    "exchanges_in_dataset": exchanges,
    "accuracy": round(accuracy, 3),
    "precision": round(precision, 3),
    "recall": round(recall, 3),
    "f1": round(f1, 3)
}

print(json.dumps(output))