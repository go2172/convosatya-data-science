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
| 1c. Scrape (Wikiquote) | `src/scrape_wikiquote.py` | W4: pulls 725 quotes from Wikiquote author pages (public API, no key) → `data/raw/quotes_wikiquote_raw.json` |
| 2. Process | `src/process.py` | Cleans text, normalizes labels, de-duplicates, keeps the 10 most frequent authors → `data/processed/quotes_processed.csv` (1,595 records after W4) |
| 3. Train & tune | `src/train.py` | TF-IDF + Logistic Regression with GridSearchCV tuning; evaluates on a held-out 20% split; saves model + metrics to `models/` |
| 4. Serve | `src/serve.py` | FastAPI inference API (W2): `/health`, `/predict` (batch, top-k candidates + confidence), empty-text validation; tested with `tests/test_serve.py` against the real trained model |
| 5. Transformer experiment | `src/finetune_bert.py` | Fine-tunes DistilBERT on the same task/split as the baseline (W3); comparison in `models/bert_metrics.json` — see "Week 3" below |
| 6. Learning curves | `src/learning_curve.py` | W4: fixed test split, W4 tuned hyperparameters held constant, train fraction 0.1 → 1.0 → `models/learning_curve.json` / `.csv` |
| 7. Error analysis | `src/error_analysis.py` | W5: per-class precision/recall/F1, confusion matrix + top confused pairs, confidence behaviour and threshold trade-off, real misclassified examples → `models/error_analysis.json`, `reports/error_analysis_w5.md` (evaluation only, no retraining) |

## Results

- **Test accuracy: 52.7%** (10-class problem; random baseline = 10%) — after W4 retraining
- **CV F1-macro: 0.427**
- Best hyperparameters (W4): char n-grams (3,5), 50,000 max features, C=0.5
- Train/test: 1,276 / 319 samples, stratified

W1 baseline (for comparison, preserved in `models/metrics_w1.json`):
43.3% test accuracy, CV F1-macro 0.338, on 716 / 180 samples. Note the
test sets are not identical — W4's held-out set is larger because the
dataset grew — so treat the +9.3 pt accuracy gain as a real improvement
on a harder, larger test set, not a paired A/B comparison. Part of the
gain may also come from Wikiquote's longer, book-passage style quotes
being more distinctive than the short DummyJSON quotes.

Authorship attribution on short quotes is a hard task — many authors write on
overlapping themes — so the model beating random by 5.3x is a solid result.
Next step: targeted data for the still-weak classes (Franklin, Abu Bakr,
Umar — see Week 5), then re-run the DistilBERT challenger with class
weighting (W6).

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/scrape.py                 # web scrape -> data/raw/quotes_raw.json
python src/scrape_api.py             # API ingest -> data/raw/quotes_api_raw.json
python src/process.py --inp data/raw/quotes_api_raw.json --top-labels 10
python src/train.py                  # tune + evaluate -> models/

# W4 data expansion + retraining + learning curves
python src/scrape_wikiquote.py   # Wikiquote -> data/raw/quotes_wikiquote_raw.json
python src/process.py --inp data/raw/quotes_raw.json data/raw/quotes_api_raw.json data/raw/quotes_wikiquote_raw.json --top-labels 10
python src/train.py              # retuned on 1,595 records -> models/
python src/learning_curve.py     # -> models/learning_curve.json/.csv

# Run the inference API (W2)
uvicorn src.serve:app --host 0.0.0.0 --port 8000

# Run the API smoke tests (real model, real HTTP test client)
python tests/test_serve.py

# W3 transformer experiment (CPU torch first: see requirements.txt)
python src/finetune_bert.py   # DistilBERT vs baseline -> models/bert_metrics.json

# W5 error analysis (evaluation only — reproduces the W4 test split exactly)
python src/error_analysis.py  # -> models/error_analysis.json, reports/error_analysis_w5.md
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

Confidence values are low in absolute terms (the classifier is 52.7%
accurate on a hard 10-class task after W4 retraining), so the top-k candidate list matters:
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

## Week 4: Data expansion, retraining, learning curves (2026-11-04)

New public source: **Wikiquote** (`src/scrape_wikiquote.py`, public
MediaWiki API, no key). Only top-level bullets from each author's
"Quotes" section were taken; Disputed, Misattributed, Attributed and
"Quotes about" sections were excluded, and quotes were capped at 400
characters to stay quote-like. Three earlier candidates did **not**
expand the existing classes and were dropped, not silently swapped in:
Quotable (`api.quotable.io`) was unreachable from this environment,
Type.fit now returns only 5 quotes, and ZenQuotes returns the same
fixed 50 quotes on every call, none by the existing 10 authors.

Raw Wikiquote yield: 725 quotes. Two pages barely yielded anything, for
documented reasons: Abu Bakr's "Quotes" section has only 2 bullets (one
over the 400-char cap, one rescued after external-link cleanup = 1
kept), and Umar's quotes live mostly in image captions rather than
bullets (1 kept). Ali's page is a redirect to "Ali", resolved with the
API's `redirects` flag (88 kept). After de-duplication against the
existing data, Wikiquote contributed **699 net new records**.

| Author | Before (W1) | After (W4) | Net new |
|--------|------------|-----------|---------|
| Albert Einstein | 44 | 288 | +244 |
| Rumi | 200 | 246 | +46 |
| Mark Twain | 51 | 209 | +158 |
| Ali ibn Abi Talib (R.A) | 130 | 206 | +76 |
| Muhammad Ali | 158 | 184 | +26 |
| Abdul Kalam | 100 | 135 | +35 |
| Friedrich Nietzsche | 40 | 113 | +73 |
| Umar ibn Al-Khattāb (R.A) | 83 | 84 | +1 |
| Benjamin Franklin | 35 | 74 | +39 |
| Abu Bakr (R.A) | 55 | 56 | +1 |
| **Total** | **896** | **1,595** | **+699** |

Retraining (`src/train.py`, same grid as W1) on the expanded data, same
80/20 stratified split recipe (now 1,276 train / 319 test):

| Model | Test accuracy | CV F1-macro | Best params |
|-------|--------------|-------------|-------------|
| W1 baseline (896 records) | 43.3% | 0.338 | char (2,5), 20k features, C=1.0 |
| **W4 retrained (1,595 records)** | **52.7%** | **0.427** | **char (3,5), 50k features, C=0.5** |

Per-class F1 on the W4 test set (319 quotes): Nietzsche 0.68, Muhammad
Ali 0.68, Rumi 0.69, Einstein 0.65 — but Franklin 0.12, Abu Bakr 0.26,
Umar 0.27, Twain 0.36. More data helped the classes it reached; the
three still-weak classes are exactly the ones Wikiquote barely added
to (Franklin +39 helped least, Abu Bakr/Umar +1 each). That is W5's
target, not a reason to claim the expansion fixed everything.

Learning curves (`src/learning_curve.py`, fixed 319-quote test set,
W4 tuned hyperparameters held constant, `models/learning_curve.json`):

| Train fraction | n_train | Train acc | Test acc | Test F1-macro |
|----------------|---------|-----------|----------|---------------|
| 0.10 | 127 | 1.000 | 0.329 | 0.265 |
| 0.25 | 319 | 0.994 | 0.414 | 0.360 |
| 0.50 | 638 | 0.972 | 0.480 | 0.415 |
| 0.75 | 957 | 0.936 | 0.489 | 0.423 |
| 1.00 | 1,276 | 0.906 | 0.527 | 0.460 |

Reading: test accuracy rises monotonically and is **still rising at
full data** (+3.8 pts from 75% → 100%), while train accuracy falls
from 1.000 to 0.906 — the model is no longer just memorizing, and more
data should still help. The 1.00 point reproduces the retrained
model's test score exactly, as designed.

**Decision: the W4 retrained model is now the serving model**
(`src/serve.py` loads `models/quote_author_classifier.joblib`; W2 smoke
tests re-run and pass against it, 2026-11-04). The W1 metrics are
preserved in `models/metrics_w1.json` for comparison.

## Week 5: Error analysis + evaluation report (2026-11-11)

`src/error_analysis.py` evaluates the W4 serving model on the exact
same 319-quote held-out test set (80/20, random_state=42, stratified) —
no retraining, evaluation only. Full numbers in
`models/error_analysis.json`; readable report in
`reports/error_analysis_w5.md`.

Per-class F1 on the test set:

| Author | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Rumi | 0.643 | 0.735 | 0.686 | 49 |
| Friedrich Nietzsche | 0.607 | 0.773 | 0.680 | 22 |
| Muhammad Ali | 0.628 | 0.730 | 0.675 | 37 |
| Albert Einstein | 0.673 | 0.638 | 0.655 | 58 |
| Ali ibn Abi Talib (R.A) | 0.548 | 0.415 | 0.472 | 41 |
| Abdul Kalam | 0.400 | 0.444 | 0.421 | 27 |
| Mark Twain | 0.500 | 0.286 | 0.364 | 42 |
| Umar ibn Al-Khattāb (R.A) | 0.308 | 0.235 | 0.267 | 17 |
| Abu Bakr (R.A) | 0.200 | 0.364 | 0.258 | 11 |
| Benjamin Franklin | 0.105 | 0.133 | 0.118 | 15 |

Weakest classes are Franklin (F1 0.12), Abu Bakr (0.26) and Umar
(0.27) — the same three Wikiquote barely added to in W4 (Franklin +39,
Abu Bakr +1, Umar +1). Twain is the surprise laggard (F1 0.36) despite
209 records: his quotes span humour, satire and aphorism, so the class
is internally diffuse.

Confusions cluster in two thematic groups: science-flavoured quotes
collapse into Einstein (Kalam → Einstein 7/27, Twain → Einstein 7/42;
Einstein → Nietzsche 6), and the religious cluster confuses among
itself (Umar → Ali 6/17, Ali → Abu Bakr 6, Ali → Umar/Franklin/Rumi 4
each) — the same collapse the W3 DistilBERT run showed, only less
extreme. Franklin's errors scatter across several authors: the model
has no stable picture of him.

Confidence carries signal but a threshold gate is not worth it: mean
top-1 confidence is 0.176 on correct predictions vs 0.146 on errors.
At a 0.2 threshold the model keeps only 14% of quotes (89% accurate on
those); at 0.3 it keeps 2% (100%). No threshold gives high accuracy
with useful coverage, so the API keeps returning top-k candidates for
relative-ranking routing, as in W2.

**Decision: no model or API change this week.** The binding constraint
is data for Franklin, Abu Bakr and Umar — W6 targets +50–100 quotes
each for those three classes, then re-runs the DistilBERT challenger
with class weighting.

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
│   ├── scrape_wikiquote.py # W4: Wikiquote ingestion (public API)
│   ├── process.py     # cleaning, normalization, label engineering
│   ├── train.py       # training, GridSearchCV tuning, evaluation
│   ├── serve.py       # FastAPI inference service (W2)
│   ├── finetune_bert.py # DistilBERT fine-tuning experiment (W3)
│   ├── learning_curve.py # W4: learning curves (fixed test split)
│   └── error_analysis.py # W5: error analysis + evaluation report
├── reports/
│   └── error_analysis_w5.md # W5: human-readable error analysis report
├── tests/
│   └── test_serve.py  # API smoke tests against the real model
├── data/
│   ├── raw/           # scraped JSON (reproducible inputs)
│   └── processed/     # model-ready CSV
├── models/
│   ├── quote_author_classifier.joblib
│   ├── metrics.json
│   ├── metrics_w1.json   # W4: preserved W1 baseline metrics
│   ├── learning_curve.json/.csv # W4: learning-curve points
│   ├── bert_metrics.json   # W3: DistilBERT vs baseline comparison
│   ├── error_analysis.json # W5: per-class + confusion + confidence analysis
│   └── bert_label_map.json # W3: label ordering for the BERT run
└── requirements.txt
```
