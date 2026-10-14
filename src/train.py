"""
Stage 3 — Model training and tuning.

Trains a text classifier that predicts a quote's primary tag from its text.
Pipeline: TF-IDF features -> Logistic Regression, with GridSearchCV
hyperparameter tuning ("fine-tuning") over n-gram range and regularization
strength. Evaluates on a held-out test split and persists the tuned model
plus metrics under models/.

Usage:
    python src/train.py [--inp data/processed/quotes_processed.csv]
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train quote author classifier")
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
    print(f"train: {len(X_train)} | test: {len(X_test)} | labels: {df['label'].nunique()}")

    param_grid = [
        {
            "tfidf__analyzer": ["word"],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__max_features": [8000, 20000],
            "clf__C": [0.5, 1.0, 2.0],
        },
        {
            "tfidf__analyzer": ["char_wb"],
            "tfidf__ngram_range": [(3, 5), (2, 5)],
            "tfidf__max_features": [20000, 50000],
            "clf__C": [0.5, 1.0, 2.0],
        },
    ]
    grid = GridSearchCV(
        build_pipeline(), param_grid, cv=3, n_jobs=-1, scoring="f1_macro"
    )
    grid.fit(X_train, y_train)
    print(f"best params: {grid.best_params_}")
    print(f"best CV f1-macro: {grid.best_score_:.4f}")

    best = grid.best_estimator_
    preds = best.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    joblib.dump(best, models_dir / "quote_author_classifier.joblib")
    (models_dir / "metrics.json").write_text(
        json.dumps(
            {
                "test_accuracy": round(float(acc), 4),
                "cv_f1_macro": round(float(grid.best_score_), 4),
                "best_params": grid.best_params_,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_labels": int(df["label"].nunique()),
            },
            indent=2,
        )
    )
    print("saved model -> models/quote_author_classifier.joblib")
    print("saved metrics -> models/metrics.json")


if __name__ == "__main__":
    main()
