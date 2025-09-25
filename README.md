# Impactree AI/ML Intern Assignment

## Approach
- Scraping: Generated ~60 PDF links using annualreports.com pattern (diverse industries/exchanges, 2020-2025). Downloaded/extracted with Playwright/requests/pdfplumber.
- Dataset: 52 balanced excerpts (26/26 labels) from reports; heuristic: Label 1 if explicit current Scope 3 reporting with 1/2.
- Model: TF-IDF vectorization + balanced Logistic Regression for binary classification.

## Assumptions
- PDF links follow observed pattern; some downloads may fail if site changes.
- Excerpts simulated from real snippets; in prod, auto-extract via regex on texts.
- 50+ unique companies across NYSE, NASDAQ, LSE, ASX.

## Instructions
1. pip install -r requirements.txt
2. playwright install
3. python scraper_script.py  # Downloads/extracts to data/
4. (Manual: Review data/texts/, create/label dataset.csv)
5. python training_script.py --validation validation.csv  # Outputs JSON