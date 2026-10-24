import random

import pytest

from server import words


def test_fallback_words_are_all_valid_and_disjoint():
    for w in words.ADJECTIVES + words.NOUNS:
        assert words.is_valid_word(w, set()), w
    assert not set(words.ADJECTIVES) & set(words.NOUNS)
    assert len(words.ADJECTIVES) >= 25 and len(words.NOUNS) >= 25


@pytest.mark.parametrize("word", ["JAM", "ZOO", "AT", "PURPLE", "red", "RE D", "CAT1", ""])
def test_invalid_words_are_rejected(word):
    assert not words.is_valid_word(word, set())


def test_used_words_are_rejected():
    assert not words.is_valid_word("CAT", {"CAT"})


def test_pair_needs_two_different_valid_words():
    assert words.valid_pair("RED", "CAT", set())
    assert not words.valid_pair("CAT", "CAT", set())
    assert not words.valid_pair("RED", "ZEBRA", set())


def test_normalize_trims_and_uppercases():
    assert words.normalize("  red ") == "RED"
    assert words.normalize(None) == ""


def test_fallback_pair_avoids_used_words():
    used = set(words.ADJECTIVES[1:]) | set(words.NOUNS[1:])
    assert words.fallback_pair(used, random.Random(0)) == (words.ADJECTIVES[0], words.NOUNS[0])
