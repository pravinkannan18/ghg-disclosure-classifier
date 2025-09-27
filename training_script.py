import pandas as pd
import json, os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Configuration
DATASET_PATH = "data/dataset.csv"
VALIDATION_PATH = "data/validation.csv"  # Optional validation file
MODEL_PATH = "data/model/model.pkl"
VECTORIZER_PATH = "data/model/vectorizer.pkl"

def load_data(path):
    """
    Load dataset from CSV.
    """
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None, None, None
    df = pd.read_csv(path)
    X = df["text_excerpt"]
    y = df["label"]
    return df, X, y

def train_model(X, y):
    """
    Train a logistic regression model using TF-IDF features.
    """
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X)
    
    # Split for internal validation
    X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate on internal validation set
    y_pred_val = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Internal Validation Accuracy: {accuracy:.2f}")
    print("Internal Validation Classification Report:")
    print(classification_report(y_val, y_pred_val))
    
    # Save model and vectorizer
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model and vectorizer saved to {MODEL_PATH} and {VECTORIZER_PATH}")
    
    return model, vectorizer

def evaluate_model(model, vectorizer, X_val, y_val):
    """
    Evaluate the model on an external validation set.
    """
    if X_val is not None and y_val is not None:
        X_val_tfidf = vectorizer.transform(X_val)
        y_pred = model.predict(X_val_tfidf)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"External Validation Accuracy: {accuracy:.2f}")
        print("External Validation Classification Report:")
        print(classification_report(y_val, y_pred))
    else:
        print("No external validation data provided.")

def get_json_output(df):
    """
    Generate JSON output with files_used_for_training and exchanges_in_dataset.
    """
    files_used = len(df)
    exchanges = df["exchange"].unique().tolist()
    output = {
        "files_used_for_training": files_used,
        "exchanges_in_dataset": exchanges
    }
    print(json.dumps(output, indent=2))

def main(validation_path=None):
    # Load training data
    train_df, X_train, y_train = load_data(DATASET_PATH)
    if train_df is None or X_train is None or y_train is None:
        return
    
    # Train model
    model, vectorizer = train_model(X_train, y_train)
    
    # Load and evaluate validation data if provided
    if validation_path and os.path.exists(validation_path):
        val_df, X_val, y_val = load_data(validation_path)
        evaluate_model(model, vectorizer, X_val, y_val)
    else:
        print("No external validation file provided. Using internal validation only.")
    
    # Print JSON output
    get_json_output(train_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GHG classifier model.")
    parser.add_argument("--validation", type=str, help="Path to validation CSV file (optional)")
    args = parser.parse_args()
    main(args.validation)