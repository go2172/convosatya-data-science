"""
Stage 1c — Data expansion (W4, new public source).

Pulls quotes from Wikiquote (https://en.wikiquote.org, public MediaWiki
API, no key required) for the existing 10 authors, and stores the raw
records as JSON under data/raw/, in the same schema as scrape.py /
scrape_api.py so all sources merge cleanly in stage 2. The training
label is the existing label for that author (authorship attribution).

Why Wikiquote, and why only the "Quotes" section: earlier W4 candidates
did not expand the existing classes — Quotable (api.quotable.io) was
unreachable from this environment, Type.fit now returns only 5 quotes,
and ZenQuotes returns the same fixed 50 quotes on every call, none by
the existing 10 authors. Wikiquote, by contrast, has a dedicated page
per author. Only top-level bullets under the "Quotes" section are
taken (nested bullets are citations, not quotes), and the Disputed /
Misattributed / Attributed / "Quotes about" sections are excluded, so
no disputed attribution enters the training set. Quotes are capped at
400 characters to stay quote-like (Wikiquote also hosts long passages).

Usage:
    python src/scrape_wikiquote.py [--out data/raw/quotes_wikiquote_raw.json]
"""

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import requests

API_URL = "https://en.wikiquote.org/w/api.php"
HEADERS = {"User-Agent": "convosatya-data-science/1.0 (data collection bot)"}
MIN_LEN = 20
MAX_LEN = 400

# Wikiquote page title -> existing training label.
AUTHORS = {
    "Albert Einstein": "Albert Einstein",
    "Mark Twain": "Mark Twain",
    "Benjamin Franklin": "Benjamin Franklin",
    "Friedrich Nietzsche": "Friedrich Nietzsche",
    "Rumi": "Rumi",
    "Muhammad Ali": "Muhammad Ali",
    "A. P. J. Abdul Kalam": "Abdul Kalam",
    "Ali ibn Abi Talib": "Ali ibn Abi Talib (R.A)",
    "Abu Bakr": "Abu Bakr (R.A)",
    "Umar": "Umar ibn Al-Khattāb (R.A)",
}


def api_get(params: dict) -> dict:
    params = {
        "action": "parse", "format": "json", "formatversion": 2,
        "redirects": 1,  # e.g. "Ali ibn Abi Talib" -> "Ali"
        **params,
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.json()["parse"]


def clean_text(text: str) -> str:
    # External links: keep the link text, drop the URL (a URL left in the
    # text would otherwise trip the "http" filter in quotes_for_page).
    # Wikitext external links use single brackets: [url link text].
    text = re.sub(r"\[https?://[^\]\s]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]+\]", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


HEADING_RE = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$")


def quotes_for_page(page: str) -> list[str]:
    """Top-level bullets from the page's 'Quotes' section only.

    One full-page fetch per author (not one call per section): section
    headings are tracked locally, and collection stops at the next
    level-2 heading, which is where the Disputed / Misattributed /
    "Quotes about" sections live.
    """
    wikitext = api_get({"page": page, "prop": "wikitext"})["wikitext"]
    quotes: list[str] = []
    in_quotes = False
    for line in wikitext.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            title = re.sub(r"<[^>]+>", "", heading.group(2)).strip().lower()
            if level == 2:
                if title == "quotes":
                    in_quotes = True
                elif in_quotes:
                    break
            continue
        if in_quotes and line.startswith("* ") and not line.startswith("**"):
            quote = clean_text(line[2:])
            if MIN_LEN <= len(quote) <= MAX_LEN and "http" not in quote:
                quotes.append(quote)
    # De-duplicate within this source, preserving order.
    return list(dict.fromkeys(quotes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest quotes via Wikiquote")
    parser.add_argument("--out", default="data/raw/quotes_wikiquote_raw.json")
    args = parser.parse_args()

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    records = []
    for page, label in AUTHORS.items():
        quotes = quotes_for_page(page)
        print(f"{page}: {len(quotes)} quotes")
        for quote in quotes:
            records.append(
                {
                    "text": quote,
                    "author": label,
                    "tags": [],
                    "label": label,  # authorship attribution task
                    "source_url": f"https://en.wikiquote.org/wiki/{page.replace(' ', '_')}",
                    "scraped_at": now,
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"saved {len(records)} raw records -> {out_path}")


if __name__ == "__main__":
    main()
