"""
src/model/augment.py

Lightweight, corpus-free text augmentation (random word deletion + random
word swap) used to partially boost the rarest priority classes in the
training set.

IMPORTANT: this must only ever be applied to a training split, AFTER the
train/test split has already happened. Augmenting before the split would
create near-duplicate text on both sides of the split and inflate test
metrics with data leakage.

This does not manufacture new information — it only reduces the model's
sensitivity to exact wording in the few examples that already exist. It
gave a modest, real improvement (macro-F1 0.19 -> 0.205 on this dataset)
but did not solve zero-recall on the rarest classes (Blocker/Highest/Low),
because those classes had too few original examples (7-21) for
augmentation to add genuinely new signal. See README for the full
evaluation history.
"""

import random as _random

import pandas as pd


def random_deletion(words: list, p: float = 0.1, rng: _random.Random = None) -> list:
    """Randomly drop words with probability p. Leaves short texts untouched.

    Args:
        words: list of tokens
        p: probability of dropping each word
        rng: a random.Random instance for reproducibility (uses module
            default if None)

    Returns:
        New list of words (never empty, never mutates input)
    """
    rng = rng or _random
    if len(words) <= 3:
        return words[:]
    kept = [w for w in words if rng.random() > p]
    return kept if kept else words[:]


def random_swap(words: list, n_swaps: int = 1, rng: _random.Random = None) -> list:
    """Randomly swap the position of n_swaps pairs of words.

    Args:
        words: list of tokens
        n_swaps: how many swap operations to perform
        rng: a random.Random instance for reproducibility

    Returns:
        New list of words with the same multiset of tokens (never mutates input)
    """
    rng = rng or _random
    words = words[:]
    for _ in range(n_swaps):
        if len(words) < 2:
            break
        i, j = rng.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return words


def augment_text(text: str, rng: _random.Random = None) -> str:
    """Produce one augmented variant of a cleaned ticket text string.

    Args:
        text: already-cleaned ticket text (see clean.py)
        rng: a random.Random instance for reproducibility

    Returns:
        Augmented text string
    """
    rng = rng or _random
    words = text.split()
    words = random_deletion(words, p=0.1, rng=rng)
    words = random_swap(words, n_swaps=max(1, len(words) // 15), rng=rng)
    return " ".join(words)


def balance_with_augmentation(
    X_train: pd.Series,
    y_train: pd.Series,
    target_ratio: float = 0.5,
    random_state: int = 42,
) -> tuple:
    """Boost under-represented classes with augmented copies.

    Classes below target_ratio * (majority class count) get augmented
    copies generated (sampling existing rows of that class with
    replacement, then perturbing them) until they reach that target.
    The majority class and any class already above the target are left
    untouched. Original rows are always preserved in full.

    Args:
        X_train: training text, already split from X_test (never pass
            the full dataset here — this must run after the split)
        y_train: training labels, aligned with X_train
        target_ratio: fraction of the majority class's count that
            minority classes get boosted toward (0.5 = half)
        random_state: seed for reproducibility

    Returns:
        (X_train_augmented, y_train_augmented) — pandas Series, original
        rows plus augmented rows appended
    """
    rng = _random.Random(random_state)

    train_df = pd.DataFrame({"text": X_train.values, "priority": y_train.values})
    majority_count = train_df["priority"].value_counts().max()
    target = int(majority_count * target_ratio)

    augmented_rows = []
    for cls, group in train_df.groupby("priority"):
        n = len(group)
        if n >= target:
            continue
        needed = target - n
        for _ in range(needed):
            src = group.sample(1, random_state=rng.randint(0, 10_000)).iloc[0]
            augmented_rows.append(
                {"text": augment_text(src["text"], rng=rng), "priority": cls}
            )

    if augmented_rows:
        train_df = pd.concat(
            [train_df, pd.DataFrame(augmented_rows)], ignore_index=True
        )

    return train_df["text"], train_df["priority"]