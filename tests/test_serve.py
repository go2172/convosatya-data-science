"""Smoke tests for the FastAPI inference service (src/serve.py).

Runs against the REAL trained model. Execute from the repo root with the
project venv active:

    .venv/bin/python tests/test_serve.py

Exit code 0 = all checks passed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from src.serve import app  # noqa: E402

SAMPLE_QUOTES = [
    "The heart has its reasons which reason knows nothing of.",
    "I have not failed. I've just found 10,000 ways that won't work.",
    "Be the change that you wish to see in the world.",
]


def main() -> int:
    with TestClient(app) as client:
        # 1. Health endpoint
        r = client.get("/health")
        assert r.status_code == 200, f"health -> {r.status_code}"
        health = r.json()
        assert health["status"] == "ok" and health["model_loaded"] is True
        assert health["n_labels"] == 10
        print(f"health ok: {health['n_labels']} labels loaded")

        # 2. Batch prediction
        r = client.post(
            "/predict", json={"texts": SAMPLE_QUOTES, "top_k": 3}
        )
        assert r.status_code == 200, f"predict -> {r.status_code}: {r.text}"
        preds = r.json()
        assert len(preds) == len(SAMPLE_QUOTES)

        for p, quote in zip(preds, SAMPLE_QUOTES):
            assert p["text"] == quote
            assert p["predicted_author"] in health["labels"], p["predicted_author"]
            assert 0.0 <= p["confidence"] <= 1.0
            assert len(p["top_candidates"]) == 3
            assert p["top_candidates"][0]["author"] == p["predicted_author"]
            # candidates sorted by descending confidence
            confs = [c["confidence"] for c in p["top_candidates"]]
            assert confs == sorted(confs, reverse=True), confs
            print(
                f"  '{quote[:45]}...' -> {p['predicted_author']} "
                f"({p['confidence']:.3f})"
            )

        # 3. Empty-string validation
        r = client.post("/predict", json={"texts": ["  "]})
        assert r.status_code == 422, f"empty text -> {r.status_code}"
        print("empty-text validation ok (422)")

        # 4. top_k clamping
        r = client.post("/predict", json={"texts": [SAMPLE_QUOTES[0]], "top_k": 10})
        assert len(r.json()[0]["top_candidates"]) == 10
        print("top_k=10 ok")

    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
