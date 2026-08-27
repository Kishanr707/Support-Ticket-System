"""tests/test_clean.py — unit tests for src/preprocessing/clean.py"""

import pandas as pd

from src.preprocessing.clean import apply_cleaning, clean_text


def test_lowercases_text():
    assert clean_text("VPN Is Down") == "vpn is down"


def test_strips_punctuation():
    assert clean_text("error: cannot connect, retrying...") == "error cannot connect retrying"


def test_preserves_placeholder_tokens():
    result = clean_text("issue with ph_ip_address on ph_technical server.")
    assert "ph_ip_address" in result
    assert "ph_technical" in result


def test_collapses_whitespace():
    assert clean_text("too    many\n\nspaces") == "too many spaces"


def test_normalizes_curly_quotes():
    result = clean_text("please refer to \u201cthe attached\u201d file")
    assert "\u201c" not in result and "\u201d" not in result


def test_handles_none_and_nan():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_handles_empty_string():
    assert clean_text("") == ""


def test_apply_cleaning_adds_column():
    df = pd.DataFrame({"ticket_text": ["VPN Is Down!", "Another, Ticket."]})
    result = apply_cleaning(df)
    assert "ticket_text_clean" in result.columns
    assert result["ticket_text_clean"].iloc[0] == "vpn is down"
    assert result["ticket_text_clean"].iloc[1] == "another ticket"