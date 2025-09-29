import requests
import os
import pdfplumber
from bs4 import BeautifulSoup
import time
import re

# Configuration
BASE_URL = "https://www.annualreports.com"
OUTPUT_PDF_DIR = "data/pdfs"
OUTPUT_TEXT_DIR = "data/text"
URLS_PATH = "data/urls.txt"

# Create directories if they don't exist
os.makedirs(OUTPUT_PDF_DIR, exist_ok=True)
os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)

def scrape_report_links():
    """
    Scrape annualreports.com for PDF download links (2020-2025), targeting diverse industries.
    Collects more than 50 links to allow for filtering.
    """
    report_links = set()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    industries = ["energy", "technology", "manufacturing", "finance", "automotive", "food", "healthcare", "retail"]
    for industry in industries:
        for page in range(1, 6):  
            url = f"{BASE_URL}/company-directory/industry/{industry}?page={page}"
            print(f"Scraping: {url}")
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
                for a in soup.find_all("a", href=True):
                    if "/Company/" in a["href"]:
                        company_url = BASE_URL + a["href"]
                        company_response = requests.get(company_url, headers=headers, timeout=10)
                        company_soup = BeautifulSoup(company_response.content, "html.parser")
                        for link in company_soup.find_all("a", href=True):
                            full_link = BASE_URL + link["href"] if link["href"].startswith("/") else link["href"]
                            if full_link.lower().endswith(".pdf") or "annual-report" in full_link.lower():
                                if any(str(year) in full_link for year in range(2020, 2026)):
                                    report_links.add(full_link)
                                    print(f"Valid link found: {full_link}")
                time.sleep(2)
            except requests.RequestException as e:
                print(f"Error scraping {url}: {e}")
    print(f"Collected {len(report_links)} unique links")
    return list(report_links)

def download_pdfs(links, headers):
    """
    Download PDFs with original names extracted from URLs.
    """
    for link in links:
        print(f"Attempting to download: {link}")
        try:
            pdf_response = requests.get(link, headers=headers, timeout=10)
            pdf_response.raise_for_status()
            filename_match = re.search(r"/([^/]+\.pdf)$", link)
            if filename_match:
                pdf_filename = filename_match.group(1)
            else:
                pdf_filename = f"report_{hash(link)}.pdf"  
            pdf_path = os.path.join(OUTPUT_PDF_DIR, pdf_filename)
            with open(pdf_path, "wb") as f:
                f.write(pdf_response.content)
            print(f"Successfully downloaded {pdf_path}")
        except requests.RequestException as e:
            print(f"Failed to download {link}: {e}")

def extract_text_from_pdfs():
    """
    Extract text from PDFs and save with original names.
    """
    for pdf_file in os.listdir(OUTPUT_PDF_DIR):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(OUTPUT_PDF_DIR, pdf_file)
            text_path = os.path.join(OUTPUT_TEXT_DIR, f"{os.path.splitext(pdf_file)[0]}.txt")
            print(f"Extracting text from {pdf_path}")
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        text += page_text + "\n"
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(text)
                print(f"Extracted text to {text_path}")
            except Exception as e:
                print(f"Error extracting text from {pdf_path}: {e}")

if __name__ == "__main__":
    print("Scraping report links from annualreports.com...")
    report_links = scrape_report_links()
    if len(report_links) < 50:
        print("Warning: Fewer than 50 links collected. Adjust scraping logic or increase page range.")
    
    with open(URLS_PATH, "w") as f:
        for link in report_links:
            f.write(f"{link}\n")
    print(f"Saved {len(report_links)} URLs to {URLS_PATH}")

    if report_links:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        print("Downloading PDFs...")
        download_pdfs(report_links, headers)
    
    if os.listdir(OUTPUT_PDF_DIR):
        print("Extracting text from PDFs...")
        extract_text_from_pdfs()
    print("Scraping and data collection complete.")