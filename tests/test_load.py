"""tests/test_load.py — unit tests for src/preprocessing/load.py"""

import pandas as pd
import pytest

from src.preprocessing.load import (
    build_labeled_dataset,
    load_priority_labels,
    load_reporter_text,
)


@pytest.fixture
def fake_utterances(tmp_path):
    path = tmp_path / "utterances.csv"
    pd.DataFrame(
        {
            "issueid": [1.0, 1.0, 2.0, 2.0, 3.0],
            "id": [10, 10, 20, 20, 30],
            "utr_seq": [0, 1, 0, 1, 0],
            "author_role": ["reporter", "reporter", "assignee", "reporter", "reporter"],
            "actionbody": ["VPN is", "not working", "looking into it", "still broken", ""],
        }
    ).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def fake_issues(tmp_path):
    path = tmp_path / "issues.csv"
    pd.DataFrame(
        {
            "id": [1.0, 2.0, 3.0, 4.0],
            "issue_priority": ["High", "Medium", "unknown", "Low"],
        }
    ).to_csv(path, index=False)
    return str(path)


def test_load_reporter_text_joins_messages_in_order(fake_utterances):
    result = load_reporter_text(fake_utterances)
    row = result[result["issueid"] == 1.0].iloc[0]
    assert row["ticket_text"] == "VPN is not working"


def test_load_reporter_text_excludes_non_reporter_messages(fake_utterances):
    result = load_reporter_text(fake_utterances)
    row = result[result["issueid"] == 2.0].iloc[0]
    assert "looking into it" not in row["ticket_text"]
    assert row["ticket_text"] == "still broken"


def test_load_reporter_text_drops_empty_text(fake_utterances):
    result = load_reporter_text(fake_utterances)
    assert 3.0 not in result["issueid"].values


def test_load_priority_labels_drops_unknown(fake_issues):
    result = load_priority_labels(fake_issues)
    assert 3.0 not in result["id"].values
    assert set(result["priority"]) == {"High", "Medium", "Low"}


def test_build_labeled_dataset_shape_and_columns(fake_utterances, fake_issues):
    df = build_labeled_dataset(fake_utterances, fake_issues)
    assert list(df.columns) == ["ticket_id", "ticket_text", "priority"]
    # issue 1 and 2 have both text and a known priority; issue 3 has no text
    # after filtering, issue 4 has no text at all
    assert len(df) == 2
    assert set(df["ticket_id"]) == {1, 2}