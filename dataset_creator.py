import os
import re
import pandas as pd

# Configuration
OUTPUT_TEXT_DIR = "data/text"
DATASET_PATH = "data/dataset.csv"
URLS_PATH = "data/urls.txt"

def extract_metadata_from_url(url):
    """
    Extract company ticker, exchange, and year from the PDF URL.
    """
    pattern = r"/HostedData/AnnualReportArchive/[a-z]/([A-Z]+)_([A-Z]+)_(\d{4})\.pdf"
    match = re.search(pattern, url)
    if match:
        ticker = match.group(1)
        exchange = match.group(2)
        year = match.group(3)
        return ticker, exchange, year
    return None, None, None

def label_text(text):
    """
    Apply labelling logic:
    - Label 1: Explicitly mentions "Scope 3" with reporting context alongside Scope 1 and 2.
    - Label 0: Mentions "Scope 1" and "Scope 2" but no "Scope 3" reporting.
    """
    text = text.lower()
    scope_1 = "scope 1" in text
    scope_2 = "scope 2" in text
    scope_3 = "scope 3" in text
    reporting_context = any(term in text for term in ["report", "reported", "reporting", "disclose", "disclosed", "disclosure"])
    future_intent = any(term in text for term in ["plan to report", "intend to report", "future", "will report", "aim to report"])
    
    if scope_1 and scope_2 and scope_3 and reporting_context and not future_intent:
        return 1
    elif scope_1 and scope_2 and not scope_3:
        return 0
    return 0  # Default to 0 if unclear

def create_dataset():
    """
    Create dataset with original filenames, aiming for 50/50 label split.
    """
    if not os.path.exists(URLS_PATH):
        print(f"Error: {URLS_PATH} not found. Run scraper_script.py first.")
        return
    
    with open(URLS_PATH, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    
    dataset = []
    text_files = [f for f in os.listdir(OUTPUT_TEXT_DIR) if f.endswith(".txt")]
    for idx, url in enumerate(urls):
        filename = [f for f in text_files if f.startswith(f"report_{idx}_") or f"report_{idx}" in f]
        if not filename:
            filename = [f for f in text_files if f.startswith(os.path.basename(url).replace(".pdf", ".txt"))]
        if filename:
            filename = filename[0]
            file_path = os.path.join(OUTPUT_TEXT_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    ticker, exchange, year = extract_metadata_from_url(url)
                    if not all([ticker, exchange, year]):
                        print(f"Warning: Incomplete metadata for {filename}, using placeholders.")
                        ticker = f"UNK_{idx}"
                        exchange = "UNK"
                        year = "2023"
                    # Excerpt: First 500 characters or up to next paragraph
                    excerpt_start = 0
                    excerpt_end = text.find("\n\n", 500) if text.find("\n\n", 500) != -1 else 500
                    text_excerpt = text[excerpt_start:excerpt_end].strip() + "..." if len(text) > 500 else text.strip()
                    label = label_text(text_excerpt)
                    dataset.append({
                        "company_name": ticker,
                        "exchange": exchange,
                        "year": year,
                        "text_excerpt": text_excerpt,
                        "label": label,
                        "original_filename": filename.replace(".txt", "")
                    })
                print(f"Processed: {filename} (Label: {label})")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Missing file for URL: {url}")

    # Ensure 50/50 label split (up to 50 entries total)
    if dataset:
        df = pd.DataFrame(dataset)
        label_0_count = len(df[df["label"] == 0])
        label_1_count = len(df[df["label"] == 1])
        total_target = min(50, len(df))  # Cap at 50 or available entries
        target_per_label = total_target // 2

        if label_0_count > target_per_label or label_1_count > target_per_label:
            # Sample to enforce 50/50
            df_0 = df[df["label"] == 0].sample(n=min(target_per_label, label_0_count), replace=False)
            df_1 = df[df["label"] == 1].sample(n=min(target_per_label, label_1_count), replace=False)
            df = pd.concat([df_0, df_1]).sample(frac=1).reset_index(drop=True)
        elif label_0_count < target_per_label or label_1_count < target_per_label:
            # Duplicate entries if needed to reach 50/50
            df_0 = df[df["label"] == 0]
            df_1 = df[df["label"] == 1]
            while len(df_0) < target_per_label and not df_0.empty:
                df_0 = pd.concat([df_0, df_0.sample(n=1, replace=True)])
            while len(df_1) < target_per_label and not df_1.empty:
                df_1 = pd.concat([df_1, df_1.sample(n=1, replace=True)])
            df = pd.concat([df_0[:target_per_label], df_1[:target_per_label]]).sample(frac=1).reset_index(drop=True)

        df.to_csv(DATASET_PATH, index=False)
        print(f"Dataset saved to {DATASET_PATH} with {len(df)} entries (Label 0: {len(df[df['label'] == 0])}, Label 1: {len(df[df['label'] == 1])})")
    else:
        print("No data to save. Check text directory and URLs.")

if __name__ == "__main__":
    print("Creating dataset from text files...")
    create_dataset()
    print("Dataset creation complete.")