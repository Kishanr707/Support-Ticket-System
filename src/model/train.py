"""
src/model/train.py

Stage 3: train a TF-IDF + LogisticRegression baseline that predicts
ticket priority from cleaned ticket text.

The vectorizer and classifier are bundled into a single sklearn Pipeline
and saved as ONE joblib file. This matters for the project's modularity
goal: predict.py only ever needs to load one artifact, so the vectorizer
and classifier can never drift out of sync with each other, and swapping
the classifier later (e.g. LinearSVC) doesn't change anything downstream.

Stopword removal happens here, in the vectorizer (stop_words='english'),
not in clean.py — see clean.py's docstring for why.

The __main__ block combines the real training split with a ChatGPT-
generated synthetic dataset (see combine.py) before fitting. This
replaced the earlier training-only-augmentation approach (augment.py)
after evaluation showed the synthetic data gives a much larger, honest
improvement on real held-out tickets (macro-F1 0.20 -> 0.38) — see
README for the full comparison. augment.py is kept in the codebase and
still works standalone; it's just no longer used in the default
training run.

CRITICAL: the real train/test split happens FIRST, and synthetic data
is only ever added to the training side. The test split stays 100%
real so evaluation numbers reflect genuine real-world performance, not
the synthetic data's own (inflated, templated) distribution.
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def split_data(
    df: pd.DataFrame,
    text_col: str = "ticket_text_clean",
    label_col: str = "priority",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified train/test split.

    Stratification matters here specifically because 'Low' priority has
    only 9 examples in the full dataset — a plain random split could
    easily leave 0 'Low' examples in the test set, making that class's
    metrics meaningless. Stratifying keeps every class proportionally
    represented in both splits.

    Args:
        df: DataFrame with text_col and label_col
        text_col: name of the cleaned text column
        label_col: name of the priority label column
        test_size: fraction held out for testing
        random_state: fixed seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[text_col]
    y = df[label_col]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_pipeline() -> Pipeline:
    """Construct the untrained TF-IDF + LogisticRegression pipeline.

    class_weight='balanced' matters here because priority classes are
    imbalanced (Medium/High dominate, Low has very few examples) —
    without it, the model could learn to mostly ignore rare classes
    and still score well on raw accuracy.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def train(X_train, y_train) -> Pipeline:
    """Fit a fresh pipeline on training data.

    Args:
        X_train: iterable of cleaned ticket text
        y_train: iterable of priority labels

    Returns:
        Fitted Pipeline
    """
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(pipeline: Pipeline, path: str = "models/priority_model.joblib") -> None:
    """Persist a fitted pipeline to disk."""
    joblib.dump(pipeline, path)


if __name__ == "__main__":
    from src.preprocessing.combine import combine_for_training

    df = pd.read_csv("data/tickets_clean.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    # Combine real training data with the synthetic dataset (training
    # side only — X_test/y_test stay 100% real, untouched).
    # ticket_id 1004298 is a known exact duplicate of a real ticket
    # found in the synthetic file and is excluded.
    X_train, y_train = combine_for_training(
        X_train,
        y_train,
        synthetic_path="data/tickets_synthetic.csv",
        exclude_ticket_ids={1004298},
    )

    pipeline = train(X_train, y_train)
    save_model(pipeline)

    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    print("Saved model -> models/priority_model.joblib")