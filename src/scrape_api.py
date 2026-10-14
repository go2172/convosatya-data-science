"""
Stage 1b — Data scraping (API source).

Pulls quotes from the free DummyJSON API (https://dummyjson.com, no key
required) and stores the raw records as JSON under data/raw/, in the same
schema as scrape.py so both sources merge cleanly in stage 2. The training
label for this source is the quote's author (authorship attribution).

Usage:
    python src/scrape_api.py [--limit N] [--out data/raw/quotes_api_raw.json]
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import requests

API_URL = "https://dummyjson.com/quotes"
HEADERS = {"User-Agent": "convosatya-data-science/1.0 (data collection bot)"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape quotes via DummyJSON API")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--out", default="data/raw/quotes_api_raw.json")
    args = parser.parse_args()

    resp = requests.get(API_URL, params={"limit": args.limit}, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    records = [
        {
            "text": item["quote"],
            "author": item["author"],
            "tags": [],
            "label": item["author"],  # authorship attribution task
            "source_url": API_URL,
            "scraped_at": now,
        }
        for item in payload.get("quotes", [])
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"saved {len(records)} raw records -> {out_path}")


if __name__ == "__main__":
    main()
