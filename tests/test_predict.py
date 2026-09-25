"""tests/test_predict.py — unit tests for src/model/predict.py"""

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import joblib

from src.model.predict import load_model, predict, predict_with_confidence
from src.model.predict import (
    load_model,
    predict,
    predict_with_confidence,
    needs_human_review,
)

@pytest.fixture
def fake_model_path(tmp_path):
    """Train a tiny throwaway pipeline on synthetic data and save it,
    so tests don't depend on the real trained model being present."""
    X = [
        "vpn is down cannot connect",
        "vpn disconnecting frequently",
        "please reset my password",
        "password reset needed",
        "printer is out of paper",
        "printer not printing",
    ]
    y = ["High", "High", "Medium", "Medium", "Low", "Low"]

    pipeline = Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(X, y)

    path = tmp_path / "fake_model.joblib"
    joblib.dump(pipeline, str(path))
    return str(path)


def test_predict_returns_known_label(fake_model_path):
    load_model.cache_clear()
    result = predict("the VPN is completely down!", model_path=fake_model_path)
    assert result in {"High", "Medium", "Low"}


def test_predict_handles_empty_string(fake_model_path):
    load_model.cache_clear()
    result = predict("", model_path=fake_model_path)
    assert result in {"High", "Medium", "Low"}


def test_predict_applies_cleaning_before_prediction(fake_model_path):
    load_model.cache_clear()
    # Messy punctuation/casing should not crash prediction
    result = predict("VPN!!! IS DOWN??? Cannot Connect...", model_path=fake_model_path)
    assert result in {"High", "Medium", "Low"}


def test_load_model_is_cached(fake_model_path):
    load_model.cache_clear()
    model_a = load_model(fake_model_path)
    model_b = load_model(fake_model_path)
    assert model_a is model_b

def test_predict_with_confidence_returns_valid_result(fake_model_path):
    load_model.cache_clear()

    priority, confidence = predict_with_confidence(
        "the VPN is completely down!",
        model_path=fake_model_path,
    )

    assert priority in {"High", "Medium", "Low"}
    assert 0.0 <= confidence <= 1.0

def test_needs_human_review_for_uncertain_rare_classes():
    assert needs_human_review("Blocker", 0.425) is True
    assert needs_human_review("Low", 0.470) is True


def test_does_not_require_review_for_confident_rare_class():
    assert needs_human_review("Blocker", 0.649) is False


def test_common_classes_do_not_require_review():
    assert needs_human_review("High", 0.484) is False
    assert needs_human_review("Medium", 0.597) is False