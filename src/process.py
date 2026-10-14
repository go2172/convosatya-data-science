"""
Stage 2 — Data processing.

Loads the raw scraped JSON, cleans and normalizes it, engineers the
training label (primary tag), and writes a model-ready CSV under
data/processed/.

Usage:
    python src/process.py [--inp data/raw/quotes_raw.json]
                          [--out data/processed/quotes_processed.csv]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# Keep only tags with enough examples to learn from.
MIN_TAG_COUNT = 5


def clean_text(text: str) -> str:
    text = text.strip().strip("\u201c\u201d\"'")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_tag(tag: str) -> str:
    return tag.strip().lower().replace("-", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Process raw quotes data")
    parser.add_argument("--inp", nargs="+", default=["data/raw/quotes_raw.json"],
                        help="one or more raw JSON files to merge")
    parser.add_argument("--out", default="data/processed/quotes_processed.csv")
    parser.add_argument("--top-labels", type=int, default=0,
                        help="keep only the K most frequent labels (0 = keep all)")
    args = parser.parse_args()

    frames = []
    for inp in args.inp:
        raw = json.loads(Path(inp).read_text(encoding="utf-8"))
        frames.append(pd.DataFrame(raw))
    df = pd.concat(frames, ignore_index=True)
    print(f"raw records: {len(df)}")

    # Drop records with no usable text, or with neither tags nor a label.
    has_label = df["label"].notna() if "label" in df.columns else False
    has_tags = df["tags"].map(lambda ts: bool(ts) if isinstance(ts, list) else False)
    df = df[df["text"].astype(bool) & (has_tags | has_label)].copy()
    df["text_clean"] = df["text"].map(clean_text)
    df = df[df["text_clean"].str.len() >= 10]

    # Normalize tags; label = explicit label when present (API source),
    # otherwise the first tag (web source).
    df["tags_norm"] = df["tags"].map(
        lambda ts: [normalize_tag(t) for t in ts] if isinstance(ts, list) else []
    )
    if "label" in df.columns and df["label"].notna().any():
        df["label"] = df["label"].where(
            df["label"].notna(),
            df["tags_norm"].map(lambda ts: ts[0] if ts else None),
        )
    else:
        df["label"] = df["tags_norm"].map(lambda ts: ts[0] if ts else None)
    df = df[df["label"].notna()]

    # De-duplicate on cleaned text.
    df = df.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)

    # Keep labels with enough support for a stable train/test split.
    counts = df["label"].value_counts()
    keep = counts[counts >= MIN_TAG_COUNT].index
    df = df[df["label"].isin(keep)]
    if args.top_labels:
        top = df["label"].value_counts().head(args.top_labels).index
        df = df[df["label"].isin(top)]
    df = df.reset_index(drop=True)

    out = df[["text_clean", "author", "label", "tags_norm", "source_url"]].rename(
        columns={"text_clean": "text"}
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"processed records: {len(out)} | labels: {out['label'].nunique()}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
