"""
Stage 6 (W4) — Learning curves for the TF-IDF + Logistic Regression model.

Answers the W4 question honestly: does more data actually help this
classifier, or has it plateaued? The held-out test split is FIXED
(80/20, random_state=42, stratified — the same split as src/train.py),
and the hyperparameters are held at the W4 tuned values
(char_wb (3,5), 50,000 max features, C=0.5 — the grid winner on the
expanded data, see models/metrics.json), so the only thing varying
across points is the fraction of the training set used. Re-tuning at
each fraction would mix two effects and make the curve unreadable.
Holding the full-data winner constant means the 1.0 point reproduces
the retrained model's test score, which is a useful consistency check.

Usage:
    python src/learning_curve.py [--inp data/processed/quotes_processed.csv]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
TEST_SIZE = 0.2
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]

# W4 tuned hyperparameters (see models/metrics.json).
BEST_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "max_features": 50000,
    "C": 0.5,
}


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    analyzer=BEST_PARAMS["analyzer"],
                    ngram_range=BEST_PARAMS["ngram_range"],
                    max_features=BEST_PARAMS["max_features"],
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=BEST_PARAMS["C"],
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Learning curves (fixed test split)")
    parser.add_argument("--inp", default="data/processed/quotes_processed.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.inp)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    print(f"full train: {len(X_train)} | test: {len(X_test)} | labels: {df['label'].nunique()}")

    points = []
    for frac in FRACTIONS:
        if frac < 1.0:
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train, train_size=frac,
                random_state=RANDOM_STATE, stratify=y_train,
            )
        else:
            X_sub, y_sub = X_train, y_train
        clf = build_pipeline()
        clf.fit(X_sub, y_sub)
        train_acc = accuracy_score(y_sub, clf.predict(X_sub))
        test_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average="macro")
        points.append(
            {
                "train_fraction": frac,
                "n_train": int(len(X_sub)),
                "train_accuracy": round(float(train_acc), 4),
                "test_accuracy": round(float(test_acc), 4),
                "test_f1_macro": round(float(test_f1), 4),
            }
        )
        print(
            f"frac {frac:.2f} | n_train {len(X_sub)} | "
            f"train_acc {train_acc:.4f} | test_acc {test_acc:.4f} | test_f1 {test_f1:.4f}"
        )

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    results = {
        "model": "TF-IDF (char_wb 3-5, 50000 features) + LogisticRegression (C=0.5, balanced)",
        "test_split": "80/20, random_state=42, stratified (same as src/train.py)",
        "n_test": int(len(X_test)),
        "n_labels": int(df["label"].nunique()),
        "points": points,
    }
    (models_dir / "learning_curve.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(points).to_csv(models_dir / "learning_curve.csv", index=False)
    print("saved metrics -> models/learning_curve.json, models/learning_curve.csv")


if __name__ == "__main__":
    main()
