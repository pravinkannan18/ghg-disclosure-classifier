import os
import re
import requests
from bs4 import BeautifulSoup
import pdfplumber
import json
from urllib.parse import urljoin

def scrape_links(base_url):
    """Dynamically scrape /Companies?exch={id} using requests + BS4 for diverse exchanges, collect PDF links for 2020-2025."""
    links_data = []
    exchanges = {'1': 'NYSE', '2': 'NASDAQ', '9': 'LSE'}  # Verified IDs for diversity
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}  # Anti-bot
    for exch_id, exch_name in exchanges.items():
        try:
            url = f"{base_url}/Companies?exch={exch_id}"
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            
            # Find company links: a with href containing /Company/
            company_links = soup.find_all('a', href=lambda h: h and '/Company/' in h)
            print(f"Found {len(company_links)} companies on {exch_name} page.")
            if company_links:
                print(f"First: {company_links[0].text.strip()} - {company_links[0]['href']}")
            
            for a in company_links[:20]:  # Limit to 20 per exchange (~60 total)
                company_name = a.text.strip()
                comp_href = a['href']
                comp_url = urljoin(base_url, comp_href)
                r_comp = requests.get(comp_url, headers=headers, timeout=10)
                r_comp.raise_for_status()
                soup_comp = BeautifulSoup(r_comp.text, 'html.parser')
                
                # Find PDF links: a ending with .pdf containing 202[0-5]
                pdf_links = soup_comp.find_all('a', href=lambda h: h and h.endswith('.pdf') and re.search(r'202[0-5]', h))
                for p in pdf_links[:3]:  # 1-3 recent per company
                    href = p['href']
                    full_link = urljoin(base_url, href)
                    year_match = re.search(r'202[0-5]', full_link)
                    year = int(year_match.group()) if year_match else 2023
                    # Industry from company page text (optional)
                    industry_match = re.search(r'(Technology|Energy|Financials|Basic Materials|Healthcare)', r_comp.text, re.IGNORECASE)
                    industry = industry_match.group(1) if industry_match else "Unknown"
                    links_data.append({
                        "link": full_link,
                        "company": company_name,
                        "exchange": exch_name,
                        "year": year,
                        "industry": industry
                    })
                    print(f"Added: {company_name} {year} ({exch_name})")
        except Exception as e:
            print(f"Error scraping {exch_name} (exch={exch_id}): {e}")
            continue
        print("---")  # Separator
    # Dedupe by company_year, cap at 60
    unique_key = lambda d: f"{d['company']}_{d['year']}"
    unique_links = {unique_key(d): d for d in links_data if d['company'] != 'Unknown'}
    links_data = list(unique_links.values())[:60]
    print(f"Scraped {len(links_data)} links from {len(set(d['company'] for d in links_data))} unique companies across {set(d['exchange'] for d in links_data)}.")
    with open("links.json", "w") as f:
        json.dump(links_data, f)
    return links_data

def download_pdfs(links_data):
    """Download PDFs to data/pdfs/."""
    os.makedirs("data/pdfs", exist_ok=True)
    successful = 0
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    for data in links_data:
        filename = f"data/pdfs/{re.sub(r'[^a-zA-Z0-9]', '_', data['company'])}_{data['year']}.pdf"
        try:
            resp = requests.get(data['link'], headers=headers, stream=True, timeout=10)
            resp.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {filename}")
            successful += 1
        except Exception as e:
            print(f"Failed {data['link']}: {e}")
    print(f"Successfully downloaded {successful}/{len(links_data)} PDFs.")

def extract_text(pdf_path):
    """Extract text from PDF using pdfplumber."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text
    except Exception as e:
        print(f"Extraction failed for {pdf_path}: {e}")
        return ""

def process_pdfs(links_data):
    """Extract text to data/texts/."""
    os.makedirs("data/texts", exist_ok=True)
    extracted = 0
    for data in links_data:
        pdf_path = f"data/pdfs/{re.sub(r'[^a-zA-Z0-9]', '_', data['company'])}_{data['year']}.pdf"
        if os.path.exists(pdf_path):
            text = extract_text(pdf_path)
            txt_path = pdf_path.replace("pdfs", "texts").replace(".pdf", ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Extracted: {txt_path}")
            extracted += 1
    print(f"Extracted {extracted}/{len(links_data)} texts.")

def main():
    base_url = "https://www.annualreports.com"
    links_data = scrape_links(base_url)
    if not links_data:
        print("No links scraped. Check site accessibility or try manual browse for selectors.")
        return
    download_pdfs(links_data)
    process_pdfs(links_data)

if __name__ == "__main__":
    # Run: pip install -r requirements.txt && python scraper_script.py
    # Uses requests + BS4 for static parsing: Finds /Company/ links, then PDFs with 202[0-5].
    # Approach: Targets exchanges for diversity; no JS needed.
    main()