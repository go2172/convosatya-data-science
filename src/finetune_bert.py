"""
Stage 5 (W3) — Transformer fine-tuning experiment.

Fine-tunes DistilBERT (distilbert-base-uncased) for the same 10-class
authorship-attribution task as the TF-IDF/LogReg baseline in src/train.py,
on the IDENTICAL train/test split (80/20, random_state=42, stratified),
so the comparison in models/bert_metrics.json and the README is fair.

The fine-tuned weights (~265 MB) are NOT committed to the repo: GitHub's
web-upload flow caps files at 25 MB and this repo is pushed through the
browser, so the weights are reproducible artifacts only. Run this script
to reproduce them; everything needed for the comparison (metrics,
per-class report, confusion matrix, label map) is committed.

Usage:
    python src/finetune_bert.py [--epochs 3] [--lr 2e-5] [--batch 8]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64


class QuoteDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True, max_length=MAX_LEN
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT vs baseline")
    parser.add_argument("--inp", default="data/processed/quotes_processed.csv")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_STATE)
    df = pd.read_csv(args.inp)
    le = LabelEncoder().fit(df["label"])
    print(f"labels ({len(le.classes_)}): {list(le.classes_)}")

    # Same split as src/train.py so the baseline comparison is fair.
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        le.transform(df["label"]),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    print(f"train: {len(X_train)} | test: {len(X_test)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    n_truncated = sum(
        len(tokenizer.encode(t)) > MAX_LEN for t in df["text"]
    )
    print(f"quotes longer than {MAX_LEN} tokens (truncated): {n_truncated}/{len(df)}")

    train_ds = QuoteDataset(X_train, y_train, tokenizer)
    test_ds = QuoteDataset(X_test, y_test, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(le.classes_)
    )
    model.config.id2label = {i: l for i, l in enumerate(le.classes_)}
    model.config.label2id = {l: i for i, l in enumerate(le.classes_)}

    training_args = TrainingArguments(
        output_dir="models/bert_checkpoints_tmp",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=16,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=27,  # ~10% of 270 total steps (90 steps/epoch x 3)
        seed=RANDOM_STATE,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds
    )
    trainer.train()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_ds), 16):
            batch = {
                k: torch.tensor(v[i : i + 16])
                for k, v in test_ds.encodings.items()
            }
            logits = model(**batch).logits
            preds.extend(torch.argmax(logits, dim=-1).tolist())
    y_true = list(y_test)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    print(f"DistilBERT test accuracy: {acc:.4f} | f1-macro: {f1:.4f}")
    print(
        classification_report(
            y_true, preds, target_names=list(le.classes_), zero_division=0
        )
    )

    baseline = json.loads(Path("models/metrics.json").read_text())
    results = {
        "model": MODEL_NAME,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch,
        "max_len": MAX_LEN,
        "test_accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_labels": int(len(le.classes_)),
        "labels": list(le.classes_),
        "baseline_tfidf_logreg": {
            "test_accuracy": baseline["test_accuracy"],
            "f1_macro": baseline["cv_f1_macro"],
            "note": "baseline f1 is CV f1-macro; accuracy is the held-out test figure",
        },
        "delta_accuracy_vs_baseline": round(
            float(acc) - float(baseline["test_accuracy"]), 4
        ),
        "classification_report": classification_report(
            y_true, preds, target_names=list(le.classes_),
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "weights_committed": False,
        "weights_note": (
            "Fine-tuned weights (~265 MB) not committed: GitHub web upload "
            "caps files at 25 MB. Re-run this script to reproduce them."
        ),
    }
    models_dir = Path("models")
    (models_dir / "bert_metrics.json").write_text(json.dumps(results, indent=2))
    (models_dir / "bert_label_map.json").write_text(
        json.dumps({l: int(i) for i, l in enumerate(le.classes_)}, indent=2)
    )
    print("saved metrics -> models/bert_metrics.json")
    print("saved label map -> models/bert_label_map.json")


if __name__ == "__main__":
    main()
