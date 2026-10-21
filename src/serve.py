"""FastAPI inference service for the ConvoSatya quote-author classifier.

Serves the trained TF-IDF + Logistic Regression pipeline saved at
models/quote_author_classifier.joblib.

Run locally:
    uvicorn src.serve:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "quote_author_classifier.joblib"

classifier = None
LABELS = []


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Quote(s) to classify")
    top_k: int = Field(
        3, ge=1, le=10, description="Number of top author candidates to return"
    )


class Prediction(BaseModel):
    text: str
    predicted_author: str
    confidence: float
    top_candidates: list[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_labels: int
    labels: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, LABELS
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}")
    classifier = joblib.load(MODEL_PATH)
    LABELS = list(classifier.classes_)
    yield


app = FastAPI(
    title="ConvoSatya Quote-Author Classifier",
    description=(
        "Authorship attribution API for short quotes. Returns the most likely "
        "of 10 authors plus confidence scores for routing/confidence thresholds."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=classifier is not None,
        n_labels=len(LABELS),
        labels=LABELS,
    )


@app.post("/predict", response_model=list[Prediction])
def predict(req: PredictRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if any(not t.strip() for t in req.texts):
        raise HTTPException(status_code=422, detail="Texts must be non-empty strings")

    proba = classifier.predict_proba(req.texts)
    top_k = min(req.top_k, len(LABELS))
    results = []
    for text, row in zip(req.texts, proba):
        order = row.argsort()[::-1][:top_k]
        candidates = [
            {"author": LABELS[i], "confidence": round(float(row[i]), 4)}
            for i in order
        ]
        results.append(
            Prediction(
                text=text,
                predicted_author=candidates[0]["author"],
                confidence=candidates[0]["confidence"],
                top_candidates=candidates,
            )
        )
    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serve:app", host="0.0.0.0", port=8000, reload=False)
