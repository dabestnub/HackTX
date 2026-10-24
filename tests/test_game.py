import random

import numpy as np

from server import words
from server.classifier import LETTERS
from server.config import Settings
from server.game import CloseRoom, FetchPair, GenerateImage, Phase, Room, Send

S = Settings(api_key=None)


def sent(outs, kind, to="any"):
    return [o.msg for o in outs if isinstance(o, Send) and o.msg["type"] == kind and (to == "any" or o.to == to)]


def onehot(letter):
    p = np.zeros(26, dtype=np.float32)
    p[LETTERS.index(letter)] = 1.0
    return p


def lobby():
    room = Room("ABCD", S, random.Random(0))
    room.join("Ann", "t1", 0.0)
    room.join("Ben", "t2", 0.0)
    return room


def start(room, pair=("RED", "CAT"), now=0.0):
    """Ready both players, supply the pair, finish the countdown. Returns the round-start outputs."""
    room.set_ready(1, True, now)
    room.set_ready(2, True, now)
    room.on_pair(*pair)
    return room.tick(now + S.countdown)


def sign(room, slot, word, t):
    """Holds each letter a little past a full hold. Returns (outputs, time after the word)."""
    outs = []
    for letter in word:
        outs += room.on_recognition(slot, onehot(letter), None, t)
        t += S.hold_seconds + 0.05
        outs += room.on_recognition(slot, onehot(letter), None, t)
        t += 0.1
    return outs, t


def test_join_assigns_slots_and_tokens():
    room = Room("ABCD", S)
    slot, outs = room.join("Ann", "t1", 0.0)
    assert slot == 1
    assert sent(outs, "room_joined", to=1) == [{"type": "room_joined", "room": "ABCD", "slot": 1, "token": "t1"}]
    assert sent(outs, "room_state")[0]["players"][0]["name"] == "Ann"
    assert room.join("Ben", "t2", 0.0)[0] == 2
    assert room.full
    assert room.slot_for_token("t2") == 2 and room.slot_for_token("nope") is None


def test_game_starts_when_both_ready():
    room = lobby()
    outs = room.set_ready(1, True, 0.0)
    assert room.phase is Phase.LOBBY and sent(outs, "room_state")
    outs = room.set_ready(2, True, 0.0)
    assert room.phase is Phase.COUNTDOWN
    assert sent(outs, "countdown") == [{"type": "countdown", "round": 1, "seconds": 3.0}]
    assert any(isinstance(o, FetchPair) for o in outs)


def test_countdown_waits_for_its_deadline():
    room = lobby()
    room.set_ready(1, True, 0.0)
    room.set_ready(2, True, 0.0)
    assert room.tick(2.9) == []
    assert room.phase is Phase.COUNTDOWN


def test_round_start_gives_each_player_a_word():
    room = lobby()
    outs = start(room, ("RED", "CAT"))
    assert room.phase is Phase.SIGNING
    first = sent(outs, "round_start", to=1)[0]
    assert first["your_word"] == "RED"  # slot 1 signs the adjective in round 1
    assert first["phrase"] == "RED CAT"
    assert first["words"] == [{"slot": 1, "word": "RED"}, {"slot": 2, "word": "CAT"}]
    assert sent(outs, "round_start", to=2)[0]["your_word"] == "CAT"
    assert sent(outs, "room_state")[0]["time_left"] == 10.0
    assert [o.used for o in outs if isinstance(o, FetchPair)] == [{"RED", "CAT"}]


def test_invalid_pair_from_muse_falls_back_to_builtin_words():
    room = lobby()
    outs = start(room, ("JAZZY", "CAT"))  # J and Z are never allowed (Q6)
    assert sent(outs, "round_start", to=1)[0]["your_word"] in words.ADJECTIVES


def test_only_the_first_valid_pair_is_kept():
    room = lobby()
    room.set_ready(1, True, 0.0)
    room.set_ready(2, True, 0.0)
    room.on_pair("red", "cat")
    room.on_pair("BIG", "DOG")
    outs = room.tick(S.countdown)
    assert sent(outs, "round_start", to=1)[0]["phrase"] == "RED CAT"


def test_ready_is_ignored_outside_the_lobby():
    room = lobby()
    start(room)
    assert room.set_ready(1, False, 4.0) == []
