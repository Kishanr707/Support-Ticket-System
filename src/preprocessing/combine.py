"""
src/preprocessing/combine.py

Combine real labeled ticket data with the ChatGPT-generated synthetic
dataset for training.

IMPORTANT: this must only ever touch the TRAINING split of the real
data, never the test split. The real held-out test set must stay
100% real, untouched, so evaluation numbers reflect genuine real-world
performance rather than the synthetic data's own (inflated, templated)
distribution. See train.py's __main__ block for how this is enforced.

The synthetic file contained one exact duplicate of a real ticket
(ticket_id 1004298) — this is excluded automatically to avoid a
trivial duplicate row.
"""

import pandas as pd


def load_synthetic_data(
    synthetic_path: str, exclude_ticket_ids: set = None
) -> pd.DataFrame:
    """Load the synthetic dataset, excluding any known duplicate real tickets.

    Args:
        synthetic_path: path to the synthetic CSV (expects columns
            ticket_id, ticket_text, priority, ticket_text_clean)
        exclude_ticket_ids: ticket_ids to drop (e.g. ones that are
            exact duplicates of real tickets)

    Returns:
        DataFrame with columns [ticket_id, ticket_text_clean, priority]
    """
    df = pd.read_csv(synthetic_path)
    if exclude_ticket_ids:
        df = df[~df["ticket_id"].isin(exclude_ticket_ids)]
    return df[["ticket_id", "ticket_text_clean", "priority"]].reset_index(drop=True)


def combine_for_training(
    X_train_real: pd.Series,
    y_train_real: pd.Series,
    synthetic_path: str,
    exclude_ticket_ids: set = None,
) -> tuple:
    """Combine the real training split with synthetic data.

    Only ever call this with the TRAINING split of real data — never
    the test split. The test split must stay real-only for honest
    evaluation.

    Args:
        X_train_real: real training text (already cleaned)
        y_train_real: real training labels
        synthetic_path: path to the synthetic CSV
        exclude_ticket_ids: ticket_ids to exclude from synthetic data
            (e.g. {1004298} for the known duplicate)

    Returns:
        (X_combined, y_combined) — pandas Series
    """
    synthetic = load_synthetic_data(synthetic_path, exclude_ticket_ids)

    X_combined = pd.concat(
        [X_train_real.reset_index(drop=True), synthetic["ticket_text_clean"]],
        ignore_index=True,
    )
    y_combined = pd.concat(
        [y_train_real.reset_index(drop=True), synthetic["priority"]],
        ignore_index=True,
    )
    return X_combined, y_combined