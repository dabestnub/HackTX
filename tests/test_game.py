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


def test_recognition_feedback_goes_to_the_signer_only():
    room = lobby()
    start(room)
    outs = room.on_recognition(1, onehot("R"), [[0.5, 0.5]] * 21, 3.0)
    assert [o.to for o in outs] == [1]
    [msg] = sent(outs, "recognition")
    assert msg == {"type": "recognition", "guess": "R", "confidence": 1.0, "hold": 0.0,
                   "accepted": False, "expected": "R", "landmarks": [[0.5, 0.5]] * 21}


def test_wrong_letters_are_ignored_without_penalty():
    room = lobby()
    start(room)
    for i in range(10):
        room.on_recognition(1, onehot("B"), None, 3.0 + i * 0.1)
    assert room.players[1].index == 0 and room.score == 0


def test_signing_both_words_requests_the_image():
    room = lobby()
    start(room)
    outs1, _ = sign(room, 1, "RED", 3.0)
    assert [m["done"] for m in sent(outs1, "progress")] == [1, 2, 3]
    assert all(m["slot"] == 1 and m["length"] == 3 for m in sent(outs1, "progress"))
    assert room.phase is Phase.SIGNING
    outs2, _ = sign(room, 2, "CAT", 3.0)
    assert room.phase is Phase.GENERATING
    assert [(o.round_no, o.phrase) for o in outs2 if isinstance(o, GenerateImage)] == [(1, "RED CAT")]
    assert sent(outs2, "image_pending") == [{"type": "image_pending", "round": 1, "phrase": "RED CAT"}]


def test_repeated_letters_need_two_holds():
    room = lobby()
    start(room, ("BIG", "BOOK"))
    outs, _ = sign(room, 2, "BOOK", 3.0)
    assert [m["done"] for m in sent(outs, "progress")] == [1, 2, 3, 4]


def test_frames_outside_signing_are_ignored():
    room = lobby()
    assert room.on_recognition(1, onehot("R"), None, 0.0) == []


def test_image_awards_points_then_reveal_then_next_round_swaps_roles():
    room = lobby()
    start(room)
    sign(room, 1, "RED", 3.0)
    _, t = sign(room, 2, "CAT", 3.0)
    outs = room.on_image(1, "/images/ABCD/1", t)
    assert sent(outs, "image_ready") == [{"type": "image_ready", "round": 1, "phrase": "RED CAT", "url": "/images/ABCD/1"}]
    assert sent(outs, "round_complete") == [{"type": "round_complete", "round": 1, "phrase": "RED CAT",
                                             "points": 100, "score": 100}]
    assert room.phase is Phase.REVEAL
    room.on_pair("BIG", "DOG")
    outs = room.tick(t + S.reveal)
    assert room.phase is Phase.COUNTDOWN and sent(outs, "countdown")[0]["round"] == 2
    outs = room.tick(t + S.reveal + S.countdown)
    assert sent(outs, "round_start", to=1)[0]["your_word"] == "DOG"  # slot 1 signs the noun in round 2


def test_stale_image_is_ignored():
    room = lobby()
    start(room)
    assert room.on_image(99, "/images/ABCD/99", 4.0) == []


def test_five_rounds_reach_500_then_game_over_and_play_again():
    room = lobby()
    room.set_ready(1, True, 0.0)
    room.set_ready(2, True, 0.0)
    t = 0.0
    pairs = [("RED", "CAT"), ("BIG", "DOG"), ("SAD", "FROG"), ("HOT", "CAKE"), ("OLD", "BOAT")]
    for i, pair in enumerate(pairs, 1):
        room.on_pair(*pair)
        t += S.countdown
        room.tick(t)
        _, t1 = sign(room, 1, room.players[1].word, t)
        _, t2 = sign(room, 2, room.players[2].word, t)
        t = max(t1, t2)
        outs = room.on_image(i, f"/images/ABCD/{i}", t)
        if i < len(pairs):
            t += S.reveal
            room.tick(t)
    assert room.phase is Phase.GAME_OVER
    [over] = sent(outs, "game_over")
    assert over["score"] == 500 and len(over["gallery"]) == 5
    assert over["gallery"][0] == {"phrase": "RED CAT", "url": "/images/ABCD/1"}
    room.play_again(1, t)
    assert room.phase is Phase.LOBBY and room.score == 0 and room.round_no == 0 and not room.used
    assert not any(p.ready for p in room.players.values())
