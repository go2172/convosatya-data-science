# ConvoSatya Data Science — Text Classification Pipeline

An end-to-end NLP pipeline built for ConvoSatya LLC: **data scraping →
data processing → model training & tuning → integration-ready artifacts**.

The pipeline collects quote data from public sources, cleans and labels it,
and trains a tuned text classifier (authorship attribution: predicting which
of 10 authors wrote a given quote). The trained model is packaged so it can
be dropped into ConvoSatya's conversational AI services for text
classification tasks such as intent/topic routing.

## Pipeline stages

| Stage | Script | What it does |
|-------|--------|--------------|
| 1a. Scrape (web) | `src/scrape.py` | Scrapes quotes, authors, and tags from quotes.toscrape.com (a practice scraping site) → `data/raw/quotes_raw.json` |
| 1b. Scrape (API) | `src/scrape_api.py` | Pulls 1,454 quotes from the free DummyJSON API → `data/raw/quotes_api_raw.json` |
| 2. Process | `src/process.py` | Cleans text, normalizes labels, de-duplicates, keeps the 10 most frequent authors → `data/processed/quotes_processed.csv` (896 records) |
| 3. Train & tune | `src/train.py` | TF-IDF + Logistic Regression with GridSearchCV tuning; evaluates on a held-out 20% split; saves model + metrics to `models/` |
| 4. Serve | `src/serve.py` | FastAPI inference API (W2): `/health`, `/predict` (batch, top-k candidates + confidence), empty-text validation; tested with `tests/test_serve.py` against the real trained model |

## Results

- **Test accuracy: 43.3%** (10-class problem; random baseline = 10%)
- **CV F1-macro: 0.338**
- Best hyperparameters: char n-grams (2,5), 20,000 max features, C=1.0
- Train/test: 716 / 180 samples, stratified

Authorship attribution on short quotes is a hard task — many authors write on
overlapping themes — so the model beating random by 4.3x is a solid baseline.
Next steps: more training data per author, transformer fine-tuning
(e.g. DistilBERT), and per-author threshold calibration.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/scrape.py                 # web scrape -> data/raw/quotes_raw.json
python src/scrape_api.py             # API ingest -> data/raw/quotes_api_raw.json
python src/process.py --inp data/raw/quotes_api_raw.json --top-labels 10
python src/train.py                  # tune + evaluate -> models/

# Run the inference API (W2)
uvicorn src.serve:app --host 0.0.0.0 --port 8000

# Run the API smoke tests (real model, real HTTP test client)
python tests/test_serve.py
```

## Serving (W2)

The trained model is exposed as a FastAPI service — `GET /health` for
liveness (model loaded, 10 labels) and `POST /predict` for batch inference
with top-k candidates and confidence scores for threshold routing.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"texts":["It is never too late to be what you might have been."],"top_k":3}'
```

Verified output (2026-10-21):

```json
[{
  "text": "It is never too late to be what you might have been.",
  "predicted_author": "Mark Twain",
  "confidence": 0.1645,
  "top_candidates": [
    {"author": "Mark Twain", "confidence": 0.1645},
    {"author": "Muhammad Ali", "confidence": 0.137},
    {"author": "Rumi", "confidence": 0.1335}
  ]
}]
```

Confidence values are low in absolute terms (the classifier is 43.3%
accurate on a hard 10-class task), so the top-k candidate list matters:
downstream callers should route on relative ranking plus a confidence
threshold rather than the raw probability alone. Empty strings are
rejected with 422.

## Integration

The trained pipeline (vectorizer + classifier in one object) is saved as
`models/quote_author_classifier.joblib` and loads with two lines:

```python
import joblib
clf = joblib.load("models/quote_author_classifier.joblib")
clf.predict(["The heart has its reasons which reason knows nothing of."])
# -> array(['Ali ibn Abi Talib (R.A)'], dtype=object)
```

`predict_proba` is available for confidence-threshold routing, and
`models/metrics.json` records the evaluation numbers for monitoring.

## Project structure

```
├── src/
│   ├── scrape.py      # web scraper (requests + BeautifulSoup)
│   ├── scrape_api.py  # API ingestion (DummyJSON)
│   ├── process.py     # cleaning, normalization, label engineering
│   ├── train.py       # training, GridSearchCV tuning, evaluation
│   └── serve.py       # FastAPI inference service (W2)
├── tests/
│   └── test_serve.py  # API smoke tests against the real model
├── data/
│   ├── raw/           # scraped JSON (reproducible inputs)
│   └── processed/     # model-ready CSV
├── models/
│   ├── quote_author_classifier.joblib
│   └── metrics.json
└── requirements.txt
```
