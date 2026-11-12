# Week 5 — Error Analysis & Evaluation Report

Evaluation of the W4 serving model (`models/quote_author_classifier.joblib`, TF-IDF + Logistic Regression) on the same held-out test set as W4: 319 quotes, 10 authors, test accuracy **52.7%** (168/319 correct, 151 errors). Split reproduced exactly (80/20, random_state=42, stratified) — no retraining, evaluation only.

## Per-class precision / recall / F1

| Author | Precision | Recall | F1 | Support | Predicted |
|---|---|---|---|---|---|
| Benjamin Franklin | 0.105 | 0.133 | 0.118 | 15 | 19 |
| Abu Bakr (R.A) | 0.200 | 0.364 | 0.258 | 11 | 20 |
| Umar ibn Al-Khattāb (R.A) | 0.308 | 0.235 | 0.267 | 17 | 13 |
| Mark Twain | 0.500 | 0.286 | 0.364 | 42 | 24 |
| Abdul Kalam | 0.400 | 0.444 | 0.421 | 27 | 30 |
| Ali ibn Abi Talib (R.A) | 0.548 | 0.415 | 0.472 | 41 | 31 |
| Albert Einstein | 0.673 | 0.638 | 0.655 | 58 | 55 |
| Muhammad Ali | 0.628 | 0.730 | 0.675 | 37 | 43 |
| Friedrich Nietzsche | 0.607 | 0.773 | 0.680 | 22 | 28 |
| Rumi | 0.643 | 0.735 | 0.686 | 49 | 56 |

## What the per-class numbers say

The three weakest classes are **Benjamin Franklin, Abu Bakr (R.A), Umar ibn Al-Khattāb (R.A)**. Franklin is the worst on every axis: low precision (0.10) and low recall (0.13) — the model both misses Franklin quotes and mislabels other authors' quotes as Franklin. Abu Bakr and Umar have the same shape at a smaller scale: tiny support (11 and 17 test quotes) and recall of 0.36 and 0.23. These are exactly the three classes Wikiquote barely added to in W4 (Franklin +39, Abu Bakr +1, Umar +1), so the data gap is still the binding constraint, not the model.

The strong classes — Rumi (F1 0.69), Friedrich Nietzsche (F1 0.68), Muhammad Ali (F1 0.68), Albert Einstein (F1 0.66) — are the ones with either the most data or the most distinctive diction. Twain (F1 0.36) is the surprise laggard despite 42 test quotes: his quotes span humour, satire and aphorism, so the class is internally diffuse.

## Top confused pairs (true → predicted)

| True author | Predicted as | Count | Share of true class |
|---|---|---|---|
| Abdul Kalam | Albert Einstein | 7 | 26% |
| Mark Twain | Albert Einstein | 7 | 17% |
| Albert Einstein | Friedrich Nietzsche | 6 | 10% |
| Ali ibn Abi Talib (R.A) | Abu Bakr (R.A) | 6 | 15% |
| Mark Twain | Abdul Kalam | 6 | 14% |
| Umar ibn Al-Khattāb (R.A) | Ali ibn Abi Talib (R.A) | 6 | 35% |
| Albert Einstein | Benjamin Franklin | 4 | 7% |
| Ali ibn Abi Talib (R.A) | Benjamin Franklin | 4 | 10% |
| Ali ibn Abi Talib (R.A) | Rumi | 4 | 10% |
| Ali ibn Abi Talib (R.A) | Umar ibn Al-Khattāb (R.A) | 4 | 10% |

The confusions are thematic, not lexical, and they cluster in two groups. First, science-flavoured quotes collapse into Einstein: 7 of Kalam's 27 test quotes (26%) and 7 of Twain's 42 (17%) are predicted Einstein, and Einstein in turn leaks 6 to Nietzsche. Second, the religious cluster confuses among itself: Umar → Ali (6, 35% of Umar's test quotes), Ali → Abu Bakr (6), Ali → Umar/Franklin/Rumi (4 each). Franklin's errors scatter across several authors instead of one rival, which is why his precision is low too: the model has no stable picture of him. The W3 DistilBERT run showed the same religious-cluster collapse, only more extreme (66/180 predictions were Ali).

## Confidence behaviour

Mean top-1 confidence is **0.176** on correct predictions vs **0.146** on errors — confidence carries real signal, but the gap is small, so a threshold can only buy accuracy at a steep coverage cost:

| Min confidence | Coverage | Accuracy on covered |
|---|---|---|
| ≥ 0.0 | 100% (319) | 52.7% |
| ≥ 0.1 | 100% (319) | 52.7% |
| ≥ 0.2 | 14% (44) | 88.6% |
| ≥ 0.3 | 2% (7) | 100.0% |
| ≥ 0.4 | 1% (2) | 100.0% |
| ≥ 0.5 | 0% (1) | 100.0% |

At a 0.3 threshold the model keeps only 2% of quotes to reach 100% accuracy; at 0.2 it keeps 14% at 89%. There is no threshold that gives high accuracy with useful coverage — the right fix is better classes, not a stricter gate. For the API, keep returning top-k candidates and let callers route on relative ranking, as W2 already does.

## Highest-confidence errors (real quotes)

- *"Love your Enemies, for they tell you your Faults."* — true: **Benjamin Franklin**, predicted **Rumi** (confidence 0.29)
- *"The best way to make your dreams come true is to wake up."* — true: **Muhammad Ali**, predicted **Abdul Kalam** (confidence 0.22)
- *"Try as you may, you don't get down as you would from a horse, you get down as you would from a house afire. You make a spectacle of yourself every time."* — true: **Mark Twain**, predicted **Rumi** (confidence 0.20)
- *"Applause Waits On Success."* — true: **Benjamin Franklin**, predicted **Abdul Kalam** (confidence 0.20)
- *"To fight against one's desires is the greatest of all fights."* — true: **Ali ibn Abi Talib (R.A)**, predicted **Abu Bakr (R.A)** (confidence 0.20)
- *"The death of a thousand worshipers is easier to bear than the death of a scholar who has knowledge of what Allah has permitted and forbidden."* — true: **Umar ibn Al-Khattāb (R.A)**, predicted **Abu Bakr (R.A)** (confidence 0.20)
- *"Allah's the Arabic term for God. Stand up for God, fight for God, work for God and do the right thing, and go the right way, things will end up in your corner."* — true: **Muhammad Ali**, predicted **Abu Bakr (R.A)** (confidence 0.19)
- *"Philosophy is empty if it isn't based on science. Science discovers, philosophy interprets."* — true: **Albert Einstein**, predicted **Friedrich Nietzsche** (confidence 0.19)
- *"Perhaps the main motive behind my isolation was my desire to escape from the demands of relationships, which I consider very difficult in comparison to making rockets. All I desired was to be true to my way of life, to uphold the science of rocketry in my country and to retire with a clean conscience."* — true: **Abdul Kalam**, predicted **Albert Einstein** (confidence 0.19)
- *"Words Are Only Painted Fire; A Look Is The Fire Itself."* — true: **Mark Twain**, predicted **Rumi** (confidence 0.19)

## Recommendations

1. **Targeted data for Franklin, Abu Bakr, Umar** — a second public source (e.g. a quotes API with author filters, or Wikiquote's 'Quotes about' sections are off-limits; try BrainyQuote-style pages via the existing scraper pattern) aimed at +50–100 quotes each for these three classes.
2. **Class-weighted loss is already on** (`class_weight='balanced'`); the remaining lever is data, not hyperparameters — W4's grid already searched it.
3. **Re-run the DistilBERT challenger after the data fix** (W3 decision) — with class weighting and more minority-class data it may stop collapsing onto the majority classes.
4. **No threshold gate in the API yet** — coverage cost is too high at any useful accuracy; keep top-k routing.

Reproduce: `python src/error_analysis.py` (full numbers in `models/error_analysis.json`).
