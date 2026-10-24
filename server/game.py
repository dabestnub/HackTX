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

    # ---- signing and images ----

    def on_recognition(self, slot: int, probs, landmarks, now: float) -> list:
        p = self.players.get(slot)
        if self.phase is not Phase.SIGNING or self.paused_at is not None or p is None or p.done:
            return []
        expected = p.word[p.index]
        guess, conf, hold, accepted = p.hold.update(expected, probs, now)
        if accepted:
            p.index += 1
        outs = [Send(slot, {"type": "recognition", "guess": guess, "confidence": round(conf, 3),
                            "hold": round(hold, 2), "accepted": accepted, "expected": expected,
                            "landmarks": landmarks})]
        if not accepted:
            return outs
        outs.append(Send(None, {"type": "progress", "slot": slot, "done": p.index, "length": len(p.word)}))
        if all(q.done for q in self.players.values()):
            self.phase = Phase.GENERATING
            self.deadline = None
            outs += [Send(None, {"type": "image_pending", "round": self.round_no, "phrase": self.phrase}),
                     self._state(now), GenerateImage(self.round_no, self.phrase)]
        return outs

    def on_image(self, round_no: int, url: str, now: float) -> list:
        if self.phase is not Phase.GENERATING or round_no != self.round_no:
            return []
        self.score += self.s.points_per_round
        self.gallery.append({"phrase": self.phrase, "url": url})
        outs = [Send(None, {"type": "image_ready", "round": round_no, "phrase": self.phrase, "url": url}),
                Send(None, {"type": "round_complete", "round": round_no, "phrase": self.phrase,
                            "points": self.s.points_per_round, "score": self.score})]
        if self.score >= self.s.target_score:
            self.phase = Phase.GAME_OVER
            self.deadline = None
            outs.append(Send(None, self._game_over()))
        else:
            self.phase = Phase.REVEAL
            self.deadline = now + self.s.reveal
        return [*outs, self._state(now)]

    def play_again(self, slot: int, now: float) -> list:
        if self.phase is not Phase.GAME_OVER:
            return []
        self.phase = Phase.LOBBY
        self.score, self.round_no, self.phrase = 0, 0, ""
        self.used, self.gallery, self.next_pair = set(), [], None
        for p in self.players.values():
            p.ready, p.word, p.index = False, "", 0
        return [self._state(now)]

    # ---- connections (Q8) ----

    def disconnect(self, slot: int, now: float) -> list:
        p = self.players.get(slot)
        if p is None or not p.connected or self.closed:
            return []
        p.connected = False
        p.hold.reset()
        if self.paused_at is None:
            self.paused_at = now
            self.close_at = now + self.s.reconnect_grace
        return [self._state(now)]

    def reconnect(self, slot: int, now: float) -> list:
        self.players[slot].connected = True
        if self.paused_at is not None and all(p.connected for p in self.players.values()):
            if self.deadline is not None:
                self.deadline += now - self.paused_at  # timers resume where they froze
            self.paused_at = self.close_at = None
        outs = [Send(slot, self._joined(slot)), self._state(now)]
        if self.phase is Phase.SIGNING:
            outs.append(Send(slot, self._round_start(slot)))
            outs += [Send(slot, {"type": "progress", "slot": p.slot, "done": p.index, "length": len(p.word)})
                     for p in self._ordered()]
        elif self.phase is Phase.GAME_OVER:
            outs.append(Send(slot, self._game_over()))
        return outs

    # ---- clock ----

    def tick(self, now: float) -> list:
        if self.closed:
            return []
        if self.paused_at is not None:  # timers freeze while someone is reconnecting (Q8)
            if now < self.close_at:
                return []
            self.closed = True
            return [Send(None, {"type": "room_closed", "reason": "partner_left"}), CloseRoom("partner_left")]
        if self.deadline is None or now < self.deadline:
            return []
        if self.phase is Phase.COUNTDOWN:
            return self._start_signing(now)
        if self.phase is Phase.SIGNING:
            return self._round_failed(now)
        if self.phase is Phase.REVEAL:
            return self._start_countdown(now)
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

    def _round_failed(self, now: float) -> list:
        missed = [p.slot for p in self._ordered() if not p.done]
        return [Send(None, {"type": "round_failed", "round": self.round_no, "reason": "timeout", "slots": missed}),
                *self._start_countdown(now)]  # 0 points; the prefetched pair starts next (Q13)

    # ---- messages ----

    def _ordered(self) -> list[Player]:
        return [self.players[s] for s in sorted(self.players)]

    def _joined(self, slot: int) -> dict:
        return {"type": "room_joined", "room": self.code, "slot": slot, "token": self.players[slot].token}

    def _round_start(self, slot: int) -> dict:
        return {"type": "round_start", "round": self.round_no, "phrase": self.phrase,
                "words": [{"slot": p.slot, "word": p.word} for p in self._ordered()],
                "your_word": self.players[slot].word, "time_limit": self.s.time_limit}

    def _game_over(self) -> dict:
        return {"type": "game_over", "score": self.score, "gallery": list(self.gallery)}

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
