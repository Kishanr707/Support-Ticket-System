"""
src/model/evaluate.py

Stage 4: evaluate a trained priority model properly.

Raw accuracy is misleading here because the priority classes are
imbalanced (Medium/High dominate, Low/Blocker are rare) — a model that
just predicts the majority class every time can score deceptively well
on accuracy alone. This module reports per-class precision/recall/F1
and a confusion matrix instead, and compares against the trivial
"always predict the majority class" baseline so the numbers have a
real point of reference.
"""

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


def majority_baseline_accuracy(y_train, y_test) -> float:
    """Accuracy of always predicting the most common training-set class.

    This is the minimum any real model needs to beat to be worth using.

    Args:
        y_train: training labels (used to find the majority class)
        y_test: test labels to score against

    Returns:
        Accuracy of the trivial majority-class baseline on y_test
    """
    majority_class = y_train.mode()[0]
    correct = (y_test == majority_class).sum()
    return correct / len(y_test)


def evaluate_model(pipeline, X_test, y_test) -> dict:
    """Run predictions and compute per-class metrics.

    Args:
        pipeline: fitted sklearn Pipeline (from train.py)
        X_test: held-out text
        y_test: held-out true labels

    Returns:
        dict with keys: 'report' (str, per-class precision/recall/F1),
        'confusion_matrix' (DataFrame), 'accuracy' (float),
        'predictions' (array)
    """
    y_pred = pipeline.predict(X_test)

    labels = sorted(y_test.unique())
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    accuracy = (y_pred == y_test.values).mean()

    return {
        "report": report,
        "confusion_matrix": cm_df,
        "accuracy": accuracy,
        "predictions": y_pred,
    }


if __name__ == "__main__":
    import joblib

    from src.model.train import split_data

    df = pd.read_csv("data/tickets_clean.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = joblib.load("models/priority_model.joblib")

    baseline = majority_baseline_accuracy(y_train, y_test)
    results = evaluate_model(pipeline, X_test, y_test)

    print(f"Majority-class baseline accuracy: {baseline:.3f}")
    print(f"Model accuracy:                   {results['accuracy']:.3f}")
    print()
    print("Per-class metrics:")
    print(results["report"])
    print()
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(results["confusion_matrix"])