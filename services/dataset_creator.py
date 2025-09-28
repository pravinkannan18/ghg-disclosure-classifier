import os
import re
import pandas as pd
import logging

# Configuration
OUTPUT_TEXT_DIR = "data/text"
DATASET_PATH = "data/dataset.csv"
MIN_ENTRIES = 250

# Set up logging
logging.basicConfig(filename="dataset_creation.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def extract_metadata(filename):
    """
    Extract company ticker, exchange, and year from filename or fallback to defaults.
    """
    match = re.search(r"([A-Z]+)_([A-Z]+)_(\d{4})\.txt", filename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    match = re.search(r"report_(\d+)_([A-Z]+)_(\d{4})\.txt", filename)
    if match:
        return f"UNK_{match.group(1)}", match.group(2), match.group(3)
    logging.warning(f"Could not extract metadata from {filename}")
    return "Unknown", "Unknown", "Unknown"

def label_excerpt(text):
    """
    Label text based on GHG emission scopes with a correct and flexible approach:
    - Label 1: Evidence of Scope 3 reporting with current context.
    - Label 0: Scope 1/2 only, or no clear Scope 3 reporting.
    """
    text = text.lower()
    # Flexible Scope detection
    has_scope1 = any(s in text for s in ["scope 1", "scope i", "ghg scope 1"])
    has_scope2 = any(s in text for s in ["scope 2", "scope ii", "ghg scope 2"])
    has_scope3 = any(s in text for s in ["scope 3", "scope iii", "ghg scope 3"])
    reporting_context = any(term in text for term in ["reported", "disclosed", "disclosure", "report", "emissions inventory", "calculated", "measured"])
    environmental_context = any(term in text for term in ["sustainability", "carbon footprint", "environmental impact", "climate strategy"])
    future_intent = any(term in text for term in ["plan to", "intend to", "will", "future", "aim to"])

    if has_scope3 and (reporting_context or environmental_context) and not future_intent:
        logging.info(f"Label 1 assigned: Scope 3 with context in {text[:50]}...")
        return 1
    elif (has_scope1 or has_scope2) and not has_scope3:
        logging.info(f"Label 0 assigned: Scope 1/2 without Scope 3 in {text[:50]}...")
        return 0
    elif not (has_scope1 or has_scope2 or has_scope3):
        logging.warning(f"Label 0 assigned (unclear): No scope terms in {text[:50]}...")
        return 0
    logging.warning(f"Label 0 assigned (unclear): Ambiguous Scope 3 context in {text[:50]}...")
    return 0  # Default to 0 for ambiguous cases

def create_dataset():
    """
    Create a balanced dataset from all text files, requiring at least 250 entries.
    """
    if not os.path.exists(OUTPUT_TEXT_DIR):
        print(f"Error: {OUTPUT_TEXT_DIR} not found. Run scraper_script.py first.")
        logging.error(f"Directory {OUTPUT_TEXT_DIR} not found.")
        return

    dataset = []
    text_files = [f for f in os.listdir(OUTPUT_TEXT_DIR) if f.endswith(".txt")]
    if len(text_files) < MIN_ENTRIES:
        print(f"Error: Only {len(text_files)} files found. At least {MIN_ENTRIES} are required. "
              "Please run scraper_script.py with more pages or companies.")
        logging.error(f"Insufficient files: {len(text_files)} < {MIN_ENTRIES}")
        return

    for filename in text_files:
        file_path = os.path.join(OUTPUT_TEXT_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            company_name, exchange, year = extract_metadata(filename)
            # Comprehensive search for relevant sections
            excerpt_start = text.lower().find("scope") if "scope" in text.lower() else text.lower().find("emissions")
            if excerpt_start == -1:
                excerpt_start = text.lower().find("sustainability")
            if excerpt_start == -1:
                excerpt_start = text.lower().find("environmental")
            if excerpt_start == -1:
                excerpt_start = text.lower().find("climate")
            if excerpt_start != -1:
                excerpt_end = text.lower().find("\n\n", excerpt_start + 500) if text.lower().find("\n\n", excerpt_start + 500) != -1 else excerpt_start + 500
                text_excerpt = text[excerpt_start:excerpt_end].strip()[:1000]  # Limit to 1000 chars
            else:
                text_excerpt = text[:1000].strip() + "..." if len(text) > 1000 else text.strip()
            label = label_excerpt(text_excerpt)
            dataset.append({
                "company_name": company_name,
                "exchange": exchange,
                "year": year,
                "text_excerpt": text_excerpt,
                "label": label,
                "original_filename": filename.replace(".txt", "")
            })
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            logging.error(f"Error processing {filename}: {str(e)}")

    if dataset:
        df = pd.DataFrame(dataset)
        # Create a balanced subset (half of total entries, or cap at reasonable size)
        total_entries = len(df)
        target_size = min(total_entries // 2, 168)  # Half of 336 or cap at 168
        label_0 = df[df["label"] == 0].sample(n=target_size, replace=False if len(df[df["label"] == 0]) >= target_size else True)
        label_1 = df[df["label"] == 1].sample(n=target_size, replace=True if len(df[df["label"] == 1]) < target_size else False)
        balanced_df = pd.concat([label_0, label_1]).sample(frac=1).reset_index(drop=True)
        balanced_df.to_csv(DATASET_PATH, index=False)
        print(f"Balanced dataset saved to {DATASET_PATH} with {len(balanced_df)} entries (Label 0: {len(label_0)}, Label 1: {len(label_1)})")
    else:
        print("No data to save. Check text directory.")
        logging.error("No data processed.")

if __name__ == "__main__":
    print("Creating dataset from all text files...")
    create_dataset()
    print("Dataset creation complete.")