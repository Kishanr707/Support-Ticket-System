"""
src/preprocessing/clean.py

Stage 2: normalize ticket text before it goes into TF-IDF.

This does normalization only (lowercasing, quote/dash normalization,
punctuation stripping, whitespace collapse). It deliberately does NOT:
  - strip ph_* placeholder tokens (e.g. ph_sql, ph_ip_address) — these
    carry real signal about ticket content and correlate with priority
  - remove stopwords — that's left to TfidfVectorizer(stop_words='english')
    in Stage 3, so it isn't duplicated here and doesn't need an nltk
    corpus download as part of this step
"""

import re

import pandas as pd

_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
    }
)

_NON_WORD_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    """Normalize a single ticket text string.

    Args:
        text: raw ticket text. Handles None/NaN gracefully.

    Returns:
        Lowercased, punctuation-stripped, whitespace-collapsed string.
        ph_* placeholder tokens are preserved intact (underscore is a
        word character, so they survive punctuation stripping).
    """
    if text is None or (isinstance(text, float)):
        return ""

    text = str(text)
    text = text.translate(_QUOTE_MAP)
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def apply_cleaning(df: pd.DataFrame, text_col: str = "ticket_text") -> pd.DataFrame:
    """Add a cleaned text column to a labeled tickets DataFrame.

    Args:
        df: DataFrame containing a raw text column (e.g. from load.py)
        text_col: name of the raw text column to clean

    Returns:
        Copy of df with an added f"{text_col}_clean" column
    """
    out = df.copy()
    out[f"{text_col}_clean"] = out[text_col].apply(clean_text)
    return out


if __name__ == "__main__":
    df = pd.read_csv("data/tickets_labeled.csv")
    df = apply_cleaning(df)
    df.to_csv("data/tickets_clean.csv", index=False)
    print(f"Cleaned {len(df)} tickets -> data/tickets_clean.csv")
    empties = (df["ticket_text_clean"].str.len() == 0).sum()
    print(f"Empty after cleaning: {empties}")