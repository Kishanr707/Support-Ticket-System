"""
src/preprocessing/load.py

Stage 1: build the labeled training table from the raw helpdesk exports.

Joins reporter-authored utterances (sample_utterances.csv) with issue
priority labels (issues.csv) to produce one clean CSV:

    ticket_id, ticket_text, priority

Only reporter messages are used as ticket_text (not assignee/other replies)
to avoid leaking post-triage information into the "live prediction" input.

Usage:
    python -m src.preprocessing.load
    (run from the project root, with data/sample_utterances.csv and
    data/issues.csv already in place)
"""

import pandas as pd


def load_reporter_text(utterances_path: str) -> pd.DataFrame:
    """Group reporter-authored utterances into one text blob per issue.

    Args:
        utterances_path: path to sample_utterances.csv

    Returns:
        DataFrame with columns [issueid, ticket_text]
    """
    utt = pd.read_csv(utterances_path)

    reporter = utt[utt["author_role"] == "reporter"].copy()
    reporter["actionbody"] = reporter["actionbody"].fillna("").astype(str)

    grouped = (
        reporter.sort_values(["issueid", "id", "utr_seq"])
        .groupby("issueid")["actionbody"]
        .apply(lambda msgs: " ".join(m.strip() for m in msgs if m.strip()))
        .reset_index()
        .rename(columns={"actionbody": "ticket_text"})
    )

    grouped = grouped[grouped["ticket_text"].str.len() > 0]
    return grouped


def load_priority_labels(issues_path: str) -> pd.DataFrame:
    """Load issue id -> priority, dropping unknown/missing priorities.

    Args:
        issues_path: path to issues.csv

    Returns:
        DataFrame with columns [id, priority]
    """
    issues = pd.read_csv(issues_path, usecols=["id", "issue_priority"])
    issues = issues.rename(columns={"issue_priority": "priority"})
    issues = issues[~issues["priority"].isin(["unknown"])]
    issues = issues.dropna(subset=["priority"])
    return issues


def build_labeled_dataset(utterances_path: str, issues_path: str) -> pd.DataFrame:
    """Join reporter text with priority labels into one training table.

    Args:
        utterances_path: path to sample_utterances.csv
        issues_path: path to issues.csv

    Returns:
        DataFrame with columns [ticket_id, ticket_text, priority]
    """
    text_df = load_reporter_text(utterances_path)
    labels_df = load_priority_labels(issues_path)

    merged = text_df.merge(
        labels_df, left_on="issueid", right_on="id", how="inner"
    )
    merged = merged.rename(columns={"issueid": "ticket_id"})
    merged = merged[["ticket_id", "ticket_text", "priority"]]
    merged["ticket_id"] = merged["ticket_id"].astype(int)

    return merged.reset_index(drop=True)


if __name__ == "__main__":
    df = build_labeled_dataset(
        utterances_path="data/sample_utterances.csv",
        issues_path="data/issues.csv",
    )
    df.to_csv("data/tickets_labeled.csv", index=False)
    print(f"Wrote {len(df)} labeled tickets to data/tickets_labeled.csv")
    print(df["priority"].value_counts())