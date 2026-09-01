"""tests/test_api.py — unit tests for src/api/main.py

Uses FastAPI's TestClient, which calls the app in-process (no real
server needs to be running to test this).
"""

import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.api.main import app


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    """Replace load_model() with a tiny throwaway pipeline so API tests
    don't depend on the real trained model file existing on disk.

    Patching load_model itself (rather than the DEFAULT_MODEL_PATH
    string) matters here: predict()'s model_path default is bound at
    import time, so patching the string after the fact would not
    actually redirect it. load_model is looked up dynamically inside
    predict(), so patching it here works correctly regardless.
    """
    X = ["vpn is down", "password reset needed", "printer out of paper"]
    y = ["High", "Medium", "Low"]
    pipeline = Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(X, y)

    monkeypatch.setattr(
        "src.model.predict.load_model", lambda model_path=None: pipeline
    )
    yield


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_priority():
    response = client.post("/predict", json={"ticket_text": "the vpn is completely down"})
    assert response.status_code == 200
    assert response.json()["priority"] in {"High", "Medium", "Low"}


def test_predict_rejects_empty_text():
    response = client.post("/predict", json={"ticket_text": ""})
    assert response.status_code == 422  # FastAPI validation error


def test_predict_rejects_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422