"""tests/test_augment.py — unit tests for src/model/augment.py"""

import random

import pandas as pd

from src.model.augment import (
    augment_text,
    balance_with_augmentation,
    random_deletion,
    random_swap,
)


def test_random_deletion_leaves_short_text_untouched():
    words = ["a", "b", "c"]
    result = random_deletion(words, p=0.9, rng=random.Random(1))
    assert result == words


def test_random_deletion_never_returns_empty():
    words = ["one", "two", "three", "four", "five"]
    result = random_deletion(words, p=1.0, rng=random.Random(1))
    assert len(result) > 0


def test_random_deletion_does_not_mutate_input():
    words = ["one", "two", "three", "four", "five"]
    original = words[:]
    random_deletion(words, p=0.5, rng=random.Random(1))
    assert words == original


def test_random_swap_preserves_word_multiset():
    words = ["alpha", "beta", "gamma", "delta"]
    result = random_swap(words, n_swaps=2, rng=random.Random(1))
    assert sorted(result) == sorted(words)


def test_random_swap_does_not_mutate_input():
    words = ["alpha", "beta", "gamma"]
    original = words[:]
    random_swap(words, n_swaps=1, rng=random.Random(1))
    assert words == original


def test_augment_text_returns_nonempty_string():
    result = augment_text("the vpn is not working today", rng=random.Random(1))
    assert isinstance(result, str)
    assert len(result) > 0


def test_balance_with_augmentation_boosts_minority_class():
    X = pd.Series(["ticket text " + str(i) for i in range(20)])
    y = pd.Series(["Medium"] * 15 + ["Low"] * 5)

    X_aug, y_aug = balance_with_augmentation(X, y, target_ratio=0.5, random_state=1)

    counts = y_aug.value_counts()
    assert counts["Low"] >= 7  # boosted toward 0.5 * 15 = 7-8
    assert counts["Medium"] == 15  # majority untouched


def test_balance_with_augmentation_preserves_all_originals():
    X = pd.Series(["a", "b", "c", "d", "e"])
    y = pd.Series(["Medium", "Medium", "Medium", "Low", "Low"])

    X_aug, y_aug = balance_with_augmentation(X, y, target_ratio=0.5, random_state=1)

    for original in ["a", "b", "c", "d", "e"]:
        assert original in X_aug.values