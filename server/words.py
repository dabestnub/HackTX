import random
import re

WORD_RE = re.compile(r"[A-IK-Y]{3,5}")  # A-Z without J and Z (Q6), 3 to 5 letters (Q7)

ADJECTIVES = [
    "RED", "BLUE", "PINK", "GOLD", "GREEN", "BLACK", "WHITE", "BIG", "TINY", "HAPPY",
    "SAD", "FAST", "SLOW", "WET", "HOT", "COLD", "SILLY", "FUNNY", "ANGRY", "SHY",
    "OLD", "CUTE", "LOUD", "SOFT", "SHINY", "BRAVE", "SPICY", "ROYAL", "TALL",
]
NOUNS = [
    "CAT", "DOG", "FISH", "BIRD", "FROG", "BEAR", "LION", "TIGER", "HORSE", "MOUSE",
    "DUCK", "COW", "PIG", "OWL", "FOX", "GOAT", "SHEEP", "CAKE", "APPLE", "TACO",
    "ROBOT", "HAT", "BOAT", "CAR", "TRAIN", "HOUSE", "TREE", "MOON", "STAR", "BOOK",
    "SHOE", "SOCK", "CHAIR", "CLOUD", "GHOST", "CROWN", "PLANT", "BUS", "KITE", "DRUM",
]


def normalize(word) -> str:
    return word.strip().upper() if isinstance(word, str) else ""  # a null Muse word must not become "NONE"


def is_valid_word(word: str, used: set[str]) -> bool:
    return WORD_RE.fullmatch(word) is not None and word not in used


def valid_pair(adjective: str, noun: str, used: set[str]) -> bool:
    return adjective != noun and is_valid_word(adjective, used) and is_valid_word(noun, used)


def fallback_pair(used: set[str], rng=random) -> tuple[str, str]:
    """A built-in pair for when Muse is off, slow, or returns bad words (Q7)."""
    # shortcut: repeats words once every fallback adjective or noun is used (29+ rounds, reachable with many timeouts); the game keeps going instead of stopping
    adjectives = [w for w in ADJECTIVES if w not in used] or ADJECTIVES
    nouns = [w for w in NOUNS if w not in used] or NOUNS
    return rng.choice(adjectives), rng.choice(nouns)
