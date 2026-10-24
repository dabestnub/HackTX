"""Pure game engine for one room: events in, messages and effects out.

No I/O and no clock. Every time-dependent method takes `now` (time.monotonic() seconds).
server/app.py delivers the Send messages and performs the FetchPair, GenerateImage and CloseRoom effects.
"""
import random
from dataclasses import dataclass
from enum import Enum

from . import words
from .acceptance import HoldTracker
from .config import Settings


class Phase(str, Enum):
    LOBBY = "lobby"
    COUNTDOWN = "countdown"
    SIGNING = "signing"
    GENERATING = "generating"
    REVEAL = "reveal"
    GAME_OVER = "game_over"


@dataclass
class Send:
    to: int | None  # slot 1 or 2; None means both players
    msg: dict


@dataclass
class FetchPair:
    used: frozenset


@dataclass
class GenerateImage:
    round_no: int
    phrase: str


@dataclass
class CloseRoom:
    reason: str


@dataclass
class Player:
    slot: int
    name: str
    token: str
    hold: HoldTracker
    connected: bool = True
    ready: bool = False
    word: str = ""
    index: int = 0

    @property
    def done(self) -> bool:
        return bool(self.word) and self.index >= len(self.word)


class Room:
    def __init__(self, code: str, settings: Settings, rng=random):
        self.code = code
        self.s = settings
        self.rng = rng
        self.players: dict[int, Player] = {}
        self.phase = Phase.LOBBY
        self.score = 0
        self.round_no = 0
        self.phrase = ""
        self.deadline: float | None = None   # when the current timed phase ends
        self.paused_at: float | None = None  # set while a player is disconnected (Q8)
        self.close_at: float | None = None
        self.used: set[str] = set()
        self.next_pair: tuple[str, str] | None = None
        self.gallery: list[dict] = []
        self.closed = False

    # ---- membership ----

    @property
    def full(self) -> bool:
        return len(self.players) == 2

    def join(self, name: str, token: str, now: float) -> tuple[int, list]:
        slot = 1 if 1 not in self.players else 2
        self.players[slot] = Player(slot, name, token, HoldTracker(self.s.hold_seconds, self.s.min_confidence))
        return slot, [Send(slot, self._joined(slot)), self._state(now)]

    def slot_for_token(self, token: str) -> int | None:
        return next((p.slot for p in self.players.values() if p.token == token), None)

    # ---- lobby and words ----

    def set_ready(self, slot: int, ready: bool, now: float) -> list:
        if self.phase is not Phase.LOBBY:
            return []
        self.players[slot].ready = ready
        if self.full and all(p.ready and p.connected for p in self.players.values()):
            return [FetchPair(frozenset(self.used)), *self._start_countdown(now)]
        return [self._state(now)]

    def on_pair(self, adjective: str, noun: str) -> list:
        adjective, noun = words.normalize(adjective), words.normalize(noun)
        if self.next_pair is None and words.valid_pair(adjective, noun, self.used):
            self.next_pair = (adjective, noun)
        return []

    # ---- clock ----

    def tick(self, now: float) -> list:
        if self.closed or self.deadline is None or now < self.deadline:
            return []
        if self.phase is Phase.COUNTDOWN:
            return self._start_signing(now)
        return []

    # ---- transitions ----

    def _start_countdown(self, now: float) -> list:
        self.phase = Phase.COUNTDOWN
        self.deadline = now + self.s.countdown
        for p in self.players.values():
            p.word, p.index = "", 0
            p.hold.reset()
        return [Send(None, {"type": "countdown", "round": self.round_no + 1, "seconds": self.s.countdown}),
                self._state(now)]

    def _start_signing(self, now: float) -> list:
        self.round_no += 1
        adjective, noun = self.next_pair or words.fallback_pair(self.used, self.rng)
        self.next_pair = None
        self.used |= {adjective, noun}
        self.phrase = f"{adjective} {noun}"
        # Slot 1 signs the adjective in odd rounds; players swap each round (Q7)
        first, second = (adjective, noun) if self.round_no % 2 else (noun, adjective)
        self.players[1].word, self.players[2].word = first, second
        self.phase = Phase.SIGNING
        self.deadline = now + self.s.time_limit
        starts = [Send(p.slot, self._round_start(p.slot)) for p in self._ordered()]
        return [*starts, self._state(now), FetchPair(frozenset(self.used))]  # prefetch the next pair

    # ---- messages ----

    def _ordered(self) -> list[Player]:
        return [self.players[s] for s in sorted(self.players)]

    def _joined(self, slot: int) -> dict:
        return {"type": "room_joined", "room": self.code, "slot": slot, "token": self.players[slot].token}

    def _round_start(self, slot: int) -> dict:
        return {"type": "round_start", "round": self.round_no, "phrase": self.phrase,
                "words": [{"slot": p.slot, "word": p.word} for p in self._ordered()],
                "your_word": self.players[slot].word, "time_limit": self.s.time_limit}

    def _time_left(self, now: float) -> float | None:
        if self.phase is not Phase.SIGNING or self.deadline is None:
            return None
        ref = self.paused_at if self.paused_at is not None else now
        return round(max(0.0, self.deadline - ref), 2)

    def _state(self, now: float) -> Send:
        return Send(None, {
            "type": "room_state", "room": self.code, "phase": self.phase.value,
            "paused": self.paused_at is not None, "score": self.score, "target": self.s.target_score,
            "round": self.round_no, "time_left": self._time_left(now),
            "players": [{"slot": p.slot, "name": p.name, "ready": p.ready, "connected": p.connected}
                        for p in self._ordered()],
        })
