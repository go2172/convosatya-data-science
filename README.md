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
| 5. Transformer experiment | `src/finetune_bert.py` | Fine-tunes DistilBERT on the same task/split as the baseline (W3); comparison in `models/bert_metrics.json` — see "Week 3" below |

## Results

- **Test accuracy: 43.3%** (10-class problem; random baseline = 10%)
- **CV F1-macro: 0.338**
- Best hyperparameters: char n-grams (2,5), 20,000 max features, C=1.0
- Train/test: 716 / 180 samples, stratified

Authorship attribution on short quotes is a hard task — many authors write on
overlapping themes — so the model beating random by 4.3x is a solid baseline.
Next steps: more training data per author (W4) and per-author threshold
calibration (W5).

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

# W3 transformer experiment (CPU torch first: see requirements.txt)
python src/finetune_bert.py   # DistilBERT vs baseline -> models/bert_metrics.json
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

## Week 3: DistilBERT fine-tuning vs the TF-IDF baseline (2026-10-28)

Following up on the baseline-evaluation question: the transformer was run
against the baseline on the **same** held-out test split (180 quotes,
random_state=42, stratified), so the numbers below are directly comparable.

Setup: `distilbert-base-uncased`, 3 epochs, lr 2e-5, batch 8, max length 64
tokens (only 5/896 quotes longer — truncated), CPU training, ~12 min.
Reproduce with `python src/finetune_bert.py` (full per-class report and
confusion matrix in `models/bert_metrics.json`).

| Model | Test accuracy | Test F1-macro | Verdict |
|-------|--------------|---------------|---------|
| TF-IDF + LogReg baseline (W1) | 43.3% | 0.341 | Still the serving model |
| DistilBERT fine-tuned (W3) | **49.4%** | 0.239 | Higher accuracy, worse balance |

The headline is mixed, and the detail explains why. DistilBERT wins on
accuracy (+6.1 pts) but **loses on F1-macro (−0.102)**: it collapses onto
the four largest classes and never predicts the other six at all
(0 predictions for Abu Bakr, Einstein, Franklin, Nietzsche, Twain, Umar).
66 of its 180 test predictions are a single class — Ali ibn Abi Talib —
absorbing the thematically overlapping religious/philosophical quotes
(Umar → Ali: 14, Abu Bakr → Ali: 10, Nietzsche → Ali: 6, Einstein → Ali: 4).
Its best classes are the well-represented ones (Rumi F1 0.74, Kalam 0.62,
Muhammad Ali 0.60).

Why: only 716 training quotes, no class weighting in the default Trainer
loss (the baseline used `class_weight="balanced"`), and 3 epochs. The
transformer is not "better" here — it is better at the majority classes
and silent on the rest.

**Decision: the TF-IDF baseline stays in production** (`src/serve.py`
unchanged). The transformer becomes the challenger again after W4 (more
data per minority author) and a class-weighted loss. Fine-tuned weights
(~265 MB) are not committed — GitHub web upload caps files at 25 MB —
but the script reproduces them exactly, and all metrics are committed.

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
│   ├── serve.py       # FastAPI inference service (W2)
│   └── finetune_bert.py # DistilBERT fine-tuning experiment (W3)
├── tests/
│   └── test_serve.py  # API smoke tests against the real model
├── data/
│   ├── raw/           # scraped JSON (reproducible inputs)
│   └── processed/     # model-ready CSV
├── models/
│   ├── quote_author_classifier.joblib
│   ├── metrics.json
│   ├── bert_metrics.json   # W3: DistilBERT vs baseline comparison
│   └── bert_label_map.json # W3: label ordering for the BERT run
└── requirements.txt
```
