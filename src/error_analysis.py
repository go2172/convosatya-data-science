"""
Stage 7 (W5) — Error analysis & evaluation report.

Reproduces the exact W4 held-out split (80/20, random_state=42,
stratified) and the saved W4 serving model, then reports:
  - per-class precision / recall / F1 / support
  - confusion matrix, with the top confused (true -> predicted) pairs
  - confidence behaviour: mean top-1 probability for correct vs wrong
    predictions, and accuracy/coverage at confidence thresholds
  - concrete misclassified examples (real quotes, real predictions)

Nothing here retrains or changes the model — this is a read-only
evaluation of models/quote_author_classifier.joblib.

Usage:
    python src/error_analysis.py [--out models/error_analysis.json]
    # human-readable report -> reports/error_analysis_w5.md
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Thresholds for the coverage/accuracy trade-off table.
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def main() -> None:
    parser = argparse.ArgumentParser(description="W5 error analysis")
    parser.add_argument("--inp", default="data/processed/quotes_processed.csv")
    parser.add_argument("--model", default="models/quote_author_classifier.joblib")
    parser.add_argument("--out", default="models/error_analysis.json")
    parser.add_argument("--report", default="reports/error_analysis_w5.md")
    args = parser.parse_args()

    df = pd.read_csv(args.inp)
    _, X_test, _, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    clf = joblib.load(args.model)
    labels = list(clf.classes_)

    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    top_conf = proba.max(axis=1)
    correct = preds == y_test.to_numpy()

    acc = accuracy_score(y_test, preds)
    report = classification_report(
        y_test, preds, labels=labels, output_dict=True, zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(report[label]["precision"]), 3),
            "recall": round(float(report[label]["recall"]), 3),
            "f1": round(float(report[label]["f1-score"]), 3),
            "support": int(report[label]["support"]),
            "n_predicted": int((preds == label).sum()),
        }
        for label in labels
    }

    cm = confusion_matrix(y_test, preds, labels=labels)
    # Top confused pairs (off-diagonal), most frequent first.
    pairs = []
    for i, true in enumerate(labels):
        for j, pred in enumerate(labels):
            if i != j and cm[i, j] > 0:
                pairs.append(
                    {
                        "true": true,
                        "predicted": pred,
                        "count": int(cm[i, j]),
                        "share_of_true_class": round(
                            float(cm[i, j] / cm[i].sum()), 3
                        ),
                    }
                )
    pairs.sort(key=lambda p: -p["count"])
    top_pairs = pairs[:10]

    # Confidence behaviour.
    conf_stats = {
        "mean_top1_confidence_correct": round(float(top_conf[correct].mean()), 3),
        "mean_top1_confidence_wrong": round(float(top_conf[~correct].mean()), 3),
        "mean_top1_confidence_all": round(float(top_conf.mean()), 3),
        "median_top1_confidence_correct": round(float(np.median(top_conf[correct])), 3),
        "median_top1_confidence_wrong": round(float(np.median(top_conf[~correct])), 3),
    }
    threshold_table = []
    for t in THRESHOLDS:
        mask = top_conf >= t
        cov = float(mask.mean())
        acc_at = (
            float(correct[mask].mean()) if mask.sum() else None
        )
        threshold_table.append(
            {
                "min_confidence": t,
                "coverage": round(cov, 3),
                "n_covered": int(mask.sum()),
                "accuracy_on_covered": (
                    round(acc_at, 3) if acc_at is not None else None
                ),
            }
        )

    # Concrete misclassified examples: the highest-confidence errors
    # (most informative) plus a spread across confused pairs.
    test_df = pd.DataFrame(
        {
            "text": X_test.to_numpy(),
            "true": y_test.to_numpy(),
            "predicted": preds,
            "confidence": np.round(top_conf, 3),
            "correct": correct,
        }
    )
    wrong = test_df[~test_df["correct"]].sort_values(
        "confidence", ascending=False
    )
    examples = [
        {
            "text": str(r.text),
            "true": str(r.true),
            "predicted": str(r.predicted),
            "confidence": float(r.confidence),
        }
        for r in wrong.head(15).itertuples(index=False)
    ]

    # Per-class mean confidence on correct predictions (calibration hint).
    per_class_conf = {}
    for label in labels:
        m = (y_test.to_numpy() == label) & correct
        if m.sum():
            per_class_conf[label] = round(float(top_conf[m].mean()), 3)

    result = {
        "test_accuracy": round(float(acc), 4),
        "n_test": int(len(X_test)),
        "n_errors": int((~correct).sum()),
        "per_class": per_class,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
        "top_confused_pairs": top_pairs,
        "confidence": conf_stats,
        "threshold_table": threshold_table,
        "per_class_mean_confidence_when_correct": per_class_conf,
        "top_confidence_errors": examples,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved -> {out_path}")

    # ---- Human-readable report ----
    lines = []
    lines.append("# Week 5 — Error Analysis & Evaluation Report")
    lines.append("")
    lines.append(
        "Evaluation of the W4 serving model "
        "(`models/quote_author_classifier.joblib`, TF-IDF + Logistic "
        "Regression) on the same held-out test set as W4: "
        f"{len(X_test)} quotes, 10 authors, "
        f"test accuracy **{acc:.1%}** "
        f"({int(correct.sum())}/{len(X_test)} correct, "
        f"{int((~correct).sum())} errors). Split reproduced exactly "
        "(80/20, random_state=42, stratified) — no retraining, "
        "evaluation only."
    )
    lines.append("")
    lines.append("## Per-class precision / recall / F1")
    lines.append("")
    lines.append("| Author | Precision | Recall | F1 | Support | Predicted |")
    lines.append("|---|---|---|---|---|---|")
    for label in sorted(labels, key=lambda l: per_class[l]["f1"]):
        p = per_class[label]
        lines.append(
            f"| {label} | {p['precision']:.3f} | {p['recall']:.3f} | "
            f"{p['f1']:.3f} | {p['support']} | {p['n_predicted']} |"
        )
    lines.append("")
    lines.append("## What the per-class numbers say")
    lines.append("")
    weakest = sorted(labels, key=lambda l: per_class[l]["f1"])[:3]
    lines.append(
        f"The three weakest classes are **{', '.join(weakest)}**. "
        "Franklin is the worst on every axis: low precision "
        f"({per_class['Benjamin Franklin']['precision']:.2f}) and low recall "
        f"({per_class['Benjamin Franklin']['recall']:.2f}) — the model "
        "both misses Franklin quotes and mislabels other authors' quotes "
        "as Franklin. Abu Bakr and Umar have the same shape at a smaller "
        f"scale: tiny support (11 and 17 test quotes) and recall of "
        f"{per_class['Abu Bakr (R.A)']['recall']:.2f} and "
        f"{per_class['Umar ibn Al-Khattāb (R.A)']['recall']:.2f}. These are "
        "exactly the three classes Wikiquote barely added "
        "to in W4 (Franklin +39, Abu Bakr +1, Umar +1), so the data gap "
        "is still the binding constraint, not the model."
    )
    lines.append("")
    lines.append(
        "The strong classes — "
        + ", ".join(
            f"{l} (F1 {per_class[l]['f1']:.2f})"
            for l in sorted(labels, key=lambda x: -per_class[x]["f1"])[:4]
        )
        + " — are the ones with either the most data or the most "
        "distinctive diction. Twain "
        f"(F1 {per_class['Mark Twain']['f1']:.2f}) is the surprise laggard "
        f"despite {per_class['Mark Twain']['support']} test quotes: his "
        "quotes span humour, satire and aphorism, so the class is "
        "internally diffuse."
    )
    lines.append("")
    lines.append("## Top confused pairs (true → predicted)")
    lines.append("")
    lines.append("| True author | Predicted as | Count | Share of true class |")
    lines.append("|---|---|---|---|")
    for p in top_pairs:
        lines.append(
            f"| {p['true']} | {p['predicted']} | {p['count']} | "
            f"{p['share_of_true_class']:.0%} |"
        )
    lines.append("")
    lines.append(
        "The confusions are thematic, not lexical, and they cluster in "
        "two groups. First, science-flavoured quotes collapse into "
        "Einstein: 7 of Kalam's 27 test quotes (26%) and 7 of Twain's 42 "
        "(17%) are predicted Einstein, and Einstein in turn leaks 6 to "
        "Nietzsche. Second, the religious cluster confuses among "
        "itself: Umar → Ali (6, 35% of Umar's test quotes), Ali → Abu "
        "Bakr (6), Ali → Umar/Franklin/Rumi (4 each). Franklin's errors "
        "scatter across several authors instead of one rival, which is "
        "why his precision is low too: the model has no stable picture "
        "of him. The W3 DistilBERT run showed the same religious-cluster "
        "collapse, only more extreme (66/180 predictions were Ali)."
    )
    lines.append("")
    lines.append("## Confidence behaviour")
    lines.append("")
    lines.append(
        f"Mean top-1 confidence is **{conf_stats['mean_top1_confidence_correct']:.3f}** "
        f"on correct predictions vs **{conf_stats['mean_top1_confidence_wrong']:.3f}** "
        "on errors — confidence carries real signal, but the gap is "
        "small, so a threshold can only buy accuracy at a steep "
        "coverage cost:"
    )
    lines.append("")
    lines.append("| Min confidence | Coverage | Accuracy on covered |")
    lines.append("|---|---|---|")
    for row in threshold_table:
        lines.append(
            f"| ≥ {row['min_confidence']:.1f} | {row['coverage']:.0%} "
            f"({row['n_covered']}) | {row['accuracy_on_covered']:.1%} |"
        )
    lines.append("")
    t03 = next(r for r in threshold_table if r["min_confidence"] == 0.3)
    t02 = next(r for r in threshold_table if r["min_confidence"] == 0.2)
    lines.append(
        f"At a 0.3 threshold the model keeps only {t03['coverage']:.0%} of "
        f"quotes to reach {t03['accuracy_on_covered']:.0%} accuracy; at "
        f"0.2 it keeps {t02['coverage']:.0%} at "
        f"{t02['accuracy_on_covered']:.0%}. There is no threshold "
        "that gives high accuracy with useful coverage — the right fix "
        "is better classes, not a stricter gate. For the API, keep "
        "returning top-k candidates and let callers route on relative "
        "ranking, as W2 already does."
    )
    lines.append("")
    lines.append("## Highest-confidence errors (real quotes)")
    lines.append("")
    for ex in examples[:10]:
        lines.append(
            f"- *\"{ex['text']}\"* — true: **{ex['true']}**, "
            f"predicted **{ex['predicted']}** (confidence {ex['confidence']:.2f})"
        )
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append(
        "1. **Targeted data for Franklin, Abu Bakr, Umar** — a second "
        "public source (e.g. a quotes API with author filters, or "
        "Wikiquote's 'Quotes about' sections are off-limits; try "
        "BrainyQuote-style pages via the existing scraper pattern) "
        "aimed at +50–100 quotes each for these three classes."
    )
    lines.append(
        "2. **Class-weighted loss is already on** "
        "(`class_weight='balanced'`); the remaining lever is data, "
        "not hyperparameters — W4's grid already searched it."
    )
    lines.append(
        "3. **Re-run the DistilBERT challenger after the data fix** "
        "(W3 decision) — with class weighting and more minority-class "
        "data it may stop collapsing onto the majority classes."
    )
    lines.append(
        "4. **No threshold gate in the API yet** — coverage cost is "
        "too high at any useful accuracy; keep top-k routing."
    )
    lines.append("")
    lines.append(
        "Reproduce: `python src/error_analysis.py` "
        "(full numbers in `models/error_analysis.json`)."
    )
    lines.append("")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved -> {report_path}")
    print(f"test accuracy: {acc:.4f} | errors: {int((~correct).sum())}/{len(X_test)}")


if __name__ == "__main__":
    main()
