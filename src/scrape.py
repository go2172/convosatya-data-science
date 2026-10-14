"""
Stage 1 — Data scraping.

Scrapes quotes, authors, and tags from the public practice site
https://quotes.toscrape.com (a site explicitly provided for scraping
practice) and stores the raw records as JSON under data/raw/.

Usage:
    python src/scrape.py [--pages N] [--out data/raw/quotes_raw.json]
"""

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://quotes.toscrape.com"
HEADERS = {"User-Agent": "convosatya-data-science/1.0 (data collection bot)"}
REQUEST_DELAY_S = 0.5


def scrape_page(page_num: int) -> list[dict]:
    """Scrape a single listing page; returns [] when the page has no quotes."""
    url = f"{BASE_URL}/page/{page_num}/"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    records = []
    for block in soup.select("div.quote"):
        text = block.select_one("span.text").get_text(strip=True)
        author = block.select_one("small.author").get_text(strip=True)
        tags = [t.get_text(strip=True) for t in block.select("div.tags a.tag")]
        records.append(
            {
                "text": text,
                "author": author,
                "tags": tags,
                "source_url": url,
                "scraped_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
    return records


def scrape_all(max_pages: int = 50) -> list[dict]:
    all_records: list[dict] = []
    for page in range(1, max_pages + 1):
        records = scrape_page(page)
        if not records:
            break
        all_records.extend(records)
        print(f"page {page}: {len(records)} quotes")
        time.sleep(REQUEST_DELAY_S)
    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape quotes dataset")
    parser.add_argument("--pages", type=int, default=50)
    parser.add_argument("--out", default="data/raw/quotes_raw.json")
    args = parser.parse_args()

    records = scrape_all(args.pages)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"saved {len(records)} raw records -> {out_path}")


if __name__ == "__main__":
    main()
