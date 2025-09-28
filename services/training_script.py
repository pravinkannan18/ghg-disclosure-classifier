import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from imblearn.over_sampling import SMOTE
import logging
import re
import joblib
import os
import json
import argparse

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    filename="../logs/training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# CLI Args
# -----------------------------
parser = argparse.ArgumentParser(description="Train and evaluate GHG disclosure classifier.")
parser.add_argument('--validation', type=str, required=True, help="Path to the validation CSV file.")
args = parser.parse_args()

# -----------------------------
# Load datasets
# -----------------------------
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
    y_val = df_val['label']
except FileNotFoundError:
    print(f"Error: {args.validation} not found.")
    logger.error(f"Validation dataset file {args.validation} not found.")
    exit(1)
except ValueError as e:
    print(f"Error: {e}")
    logger.error(f"Invalid validation dataset format: {e}")
    exit(1)

y_train = df_train['label']

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', str(text).lower().strip())
    text = re.sub(r'[^\w\s]', '', text)
    return text

df_train['text_excerpt'] = df_train['text_excerpt'].apply(preprocess_text)
df_val['text_excerpt'] = df_val['text_excerpt'].apply(preprocess_text)

# -----------------------------
# TF-IDF features
# -----------------------------
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['text_excerpt'])
X_val_tfidf = tfidf_vectorizer.transform(df_val['text_excerpt'])

# -----------------------------
# DistilBERT embeddings with PCA
# -----------------------------
def get_bert_embeddings(texts, batch_size=16):
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

pca = PCA(n_components=20)
X_train_bert_res_pca = pca.fit_transform(X_train_bert)
X_val_bert_res_pca = pca.transform(X_val_bert)

# -----------------------------
# Handle class imbalance
# -----------------------------
# Custom sampling strategy to avoid ratio issues
minority_class_size = class_dist.get(1, 0)  # Get the number of Label 1 samples
majority_class_size = class_dist.get(0, 0)  # Get the number of Label 0 samples
target_minority_size = min(majority_class_size, minority_class_size * 2)  # Limit to 2x minority or majority size
smote_tfidf = SMOTE(sampling_strategy={1: target_minority_size}, random_state=42)
X_train_tfidf_res, y_train_res = smote_tfidf.fit_resample(X_train_tfidf, y_train)

smote_bert = SMOTE(sampling_strategy={1: target_minority_size}, random_state=42)
X_train_bert_res, y_train_res = smote_bert.fit_resample(X_train_bert_res_pca, y_train)

# -----------------------------
# Models
# -----------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=0.0001, class_weight='balanced'),
    'SVM': SVC(kernel='linear', random_state=42, probability=True, C=0.0001, class_weight='balanced'),
    'DistilBERT': RidgeClassifier(random_state=42, alpha=0.01)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    if name == 'DistilBERT':
        model.fit(X_train_bert_res, y_train_res)
        y_pred_val = model.predict(X_val_bert_res_pca)
        X_train_used = X_train_bert_res
    else:
        model.fit(X_train_tfidf_res, y_train_res)
        y_pred_val = model.predict(X_val_tfidf)
        X_train_used = X_train_tfidf_res

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_used, y_train_res, cv=cv, scoring='f1_macro')
    print(f"{name} - Cross-validation F1 Macro: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
    logger.info(f"{name} - CV F1 Macro: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")

    # Validation
    print(f"\n{name} Final Validation Classification Report:")
    print(classification_report(y_val, y_pred_val, target_names=['Label 0', 'Label 1'], zero_division=0))
    print(f"Final Validation Confusion Matrix:\n{confusion_matrix(y_val, y_pred_val)}")

    logger.info(f"{name} Validation Report: {classification_report(y_val, y_pred_val, output_dict=True, zero_division=0)}")
    logger.info(f"{name} Validation Confusion Matrix: {confusion_matrix(y_val, y_pred_val)}")

    results[name] = {'y_val': y_val, 'y_pred_val': y_pred_val}

# -----------------------------
# Save best model
# -----------------------------
best_model_name = max(
    results,
    key=lambda k: classification_report(results[k]['y_val'], results[k]['y_pred_val'], output_dict=True, zero_division=0)['macro avg']['f1-score']
)
best_model = models[best_model_name]
best_preds = results[best_model_name]['y_pred_val']

joblib.dump(best_model, f"../models/best_model_{best_model_name.replace(' ', '_')}.pkl")
joblib.dump(tfidf_vectorizer, '../models/tfidf_vectorizer.pkl')
logger.info(f"Saved best model: {best_model_name}")

# -----------------------------
# Compute final metrics
# -----------------------------
accuracy = accuracy_score(y_val, best_preds)
precision = precision_score(y_val, best_preds, average="macro", zero_division=0)
recall = recall_score(y_val, best_preds, average="macro", zero_division=0)
f1 = f1_score(y_val, best_preds, average="macro", zero_division=0)

# -----------------------------
# JSON output (required format)
# -----------------------------
files_used = len(df_train['original_filename'].unique())
exchanges = df_train['exchange'].dropna().unique().tolist()

output = {
    "files_used_for_training": files_used,
    "exchanges_in_dataset": exchanges,
    "accuracy": round(accuracy, 2),
    "precision": round(precision, 2),
    "recall": round(recall, 2),
    "f1": round(f1, 2)
}

print(json.dumps(output))