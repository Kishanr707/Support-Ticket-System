"""tests/test_combine.py — unit tests for src/preprocessing/combine.py"""

import pandas as pd
import pytest

from src.preprocessing.combine import combine_for_training, load_synthetic_data


@pytest.fixture
def fake_synthetic_csv(tmp_path):
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(
        {
            "ticket_id": [9001, 9002, 9003],
            "ticket_text": ["raw a", "raw b", "raw c"],
            "priority": ["High", "Medium", "Low"],
            "ticket_text_clean": ["clean a", "clean b", "clean c"],
        }
    ).to_csv(path, index=False)
    return str(path)


def test_load_synthetic_data_returns_expected_columns(fake_synthetic_csv):
    result = load_synthetic_data(fake_synthetic_csv)
    assert list(result.columns) == ["ticket_id", "ticket_text_clean", "priority"]
    assert len(result) == 3


def test_load_synthetic_data_excludes_given_ids(fake_synthetic_csv):
    result = load_synthetic_data(fake_synthetic_csv, exclude_ticket_ids={9002})
    assert 9002 not in result["ticket_id"].values
    assert len(result) == 2


def test_combine_for_training_concatenates_real_and_synthetic(fake_synthetic_csv):
    X_train_real = pd.Series(["real x", "real y"])
    y_train_real = pd.Series(["Medium", "High"])

    X_combined, y_combined = combine_for_training(
        X_train_real, y_train_real, fake_synthetic_csv
    )

    assert len(X_combined) == 5  # 2 real + 3 synthetic
    assert len(y_combined) == 5
    assert "real x" in X_combined.values
    assert "clean a" in X_combined.values


def test_combine_for_training_respects_exclusion(fake_synthetic_csv):
    X_train_real = pd.Series(["real x"])
    y_train_real = pd.Series(["Medium"])

    X_combined, y_combined = combine_for_training(
        X_train_real, y_train_real, fake_synthetic_csv, exclude_ticket_ids={9001}
    )

    assert len(X_combined) == 3  # 1 real + 2 synthetic (9001 excluded)