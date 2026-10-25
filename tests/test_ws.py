import asyncio
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.app import FRAME_GAP, Hub, create_app
from server.classifier import LETTERS
from server.config import Settings
from server.game import Phase

FAST = Settings(countdown=0.2, hold_seconds=0.05, reveal=0.2, api_key=None)


class LetterStub:
    """Only tests interpret frame bytes as letters; production always recognizes JPEGs."""

    def process(self, jpeg):
        probs = np.zeros(26, dtype=np.float32)
        probs[LETTERS.index(jpeg.decode())] = 1.0
        return SimpleNamespace(probs=probs, landmarks=None)

    def close(self):
        pass


def recv(ws, kind):
    while True:
        msg = ws.receive_json()
        if msg["type"] == kind:
            return msg


def sign(ws, word):
    for letter in word:
        while True:
            ws.send_bytes(letter.encode())
            if recv(ws, "recognition")["accepted"]:
                break


def test_full_round_over_the_websocket():
    app = create_app(FAST, make_recognizer=LetterStub)
    with TestClient(app) as client, client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        a.send_json({"type": "create_room", "name": "Ann"})
        joined = recv(a, "room_joined")
        assert joined["slot"] == 1 and joined["token"]
        b.send_json({"type": "join_room", "room": joined["room"].lower(), "name": "Ben"})
        assert recv(b, "room_joined")["slot"] == 2
        a.send_json({"type": "ready", "ready": True})
        b.send_json({"type": "ready", "ready": True})
        word_a = recv(a, "round_start")["your_word"]
        word_b = recv(b, "round_start")["your_word"]
        sign(a, word_a)
        sign(b, word_b)
        ready = recv(a, "image_ready")
        assert ready["phrase"] == f"{word_a} {word_b}"
        assert recv(a, "round_complete")["score"] == 100
        image = client.get(ready["url"])
        assert image.status_code == 200
        assert image.headers["content-type"].startswith("image/svg+xml")
        assert client.get("/healthz").json() == {"ok": True, "rooms": 1}


def test_join_unknown_room_errors():
    with TestClient(create_app(FAST, make_recognizer=LetterStub)) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "join_room", "room": "QQQQ", "name": "X"})
        assert ws.receive_json()["code"] == "room_not_found"


@pytest.mark.parametrize("first", [{"type": "ready"}, [], "broken", b"JPEG"])
def test_first_message_must_join(first):
    with TestClient(create_app(FAST, make_recognizer=LetterStub)) as client, client.websocket_connect("/ws") as ws:
        if isinstance(first, bytes):
            ws.send_bytes(first)
        elif first == "broken":
            ws.send_text(first)
        else:
            ws.send_json(first)
        assert ws.receive_json()["code"] == "join_first"


def test_rejoin_with_token_restores_the_slot():
    with TestClient(create_app(FAST, make_recognizer=LetterStub)) as client:
        with client.websocket_connect("/ws") as a:
            a.send_json({"type": "create_room", "name": "Ann"})
            joined = recv(a, "room_joined")
        with client.websocket_connect("/ws") as again:
            again.send_json({"type": "rejoin", "room": joined["room"], "token": joined["token"]})
            assert recv(again, "room_joined")["slot"] == 1
            assert recv(again, "room_state")["players"][0]["connected"] is True
        with client.websocket_connect("/ws") as bad:
            bad.send_json({"type": "rejoin", "room": joined["room"], "token": "wrong"})
            assert bad.receive_json()["code"] == "bad_token"


def test_full_room_rejects_third_player_and_http_allows_vercel():
    with TestClient(create_app(FAST, make_recognizer=LetterStub)) as client:
        with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b, client.websocket_connect("/ws") as c:
            a.send_json({"type": "create_room"})
            code = recv(a, "room_joined")["room"]
            b.send_json({"type": "join_room", "room": code})
            recv(b, "room_joined")
            c.send_json({"type": "join_room", "room": code})
            assert c.receive_json()["code"] == "room_full"
            a.send_text("[]")
            assert recv(a, "error")["code"] == "bad_message"
            a.send_json({"type": "typed_letter", "letter": "A"})
            assert recv(a, "error")["code"] == "unknown_type"
        assert client.get("/images/QQQQ/1").status_code == 404
        response = client.get("/healthz", headers={"Origin": "https://game.vercel.app"})
        assert response.headers["access-control-allow-origin"] == "*"


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.005)


def signing(hub):
    live, conn, _ = hub.join({"type": "create_room"}, SimpleNamespace())
    hub.join({"type": "join_room", "room": live.room.code}, SimpleNamespace())
    now = time.monotonic()
    hub.dispatch(live, live.room.set_ready(1, True, now))
    hub.dispatch(live, live.room.set_ready(2, True, now))
    live.room.on_pair("RED", "CAT")
    hub.dispatch(live, live.room.tick(now + hub.s.countdown))
    while not conn.outbox.empty():
        conn.outbox.get_nowait()
    return live, conn


class GatedRecognizer(LetterStub):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.calls = []

    def process(self, jpeg):
        self.calls.append((jpeg, time.monotonic(), threading.get_ident()))
        self.started.set()
        assert self.release.wait(2), "test must release CPU work"
        return super().process(jpeg)

    def close(self):
        self.closed.set()


def test_newest_frame_is_bounded_throttled_and_cpu_runs_off_loop():
    async def scenario():
        rec = GatedRecognizer()
        hub = Hub(FAST, lambda: rec)
        live, conn = signing(hub)
        worker = asyncio.create_task(hub.recognize(live, conn))
        try:
            hub.on_frame(live, conn, b"R")
            await eventually(rec.started.is_set)
            for frame in (b"A", b"B", b"C"):
                hub.on_frame(live, conn, frame)
            rec.release.set()
            await eventually(lambda: len(rec.calls) == 2)
            assert [call[0] for call in rec.calls] == [b"R", b"C"]
            assert rec.calls[1][1] - rec.calls[0][1] >= FRAME_GAP - 0.005
            assert all(call[2] != threading.get_ident() for call in rec.calls)
        finally:
            rec.release.set()
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await hub.shutdown()
        assert rec.closed.is_set()
    asyncio.run(scenario())


@pytest.mark.parametrize("transition", ["replacement", "round", "restart", "deadline", "pause"])
def test_pending_recognition_cannot_cross_a_lifecycle_boundary(transition):
    async def scenario():
        rec = GatedRecognizer()
        hub = Hub(FAST, lambda: rec)
        live, conn = signing(hub)
        live.room.players[1].hold.since = time.monotonic() - 1
        worker = asyncio.create_task(hub.recognize(live, conn))
        try:
            hub.on_frame(live, conn, b"R")
            await eventually(rec.started.is_set)
            if transition == "replacement":
                hub.join({"type": "rejoin", "room": live.room.code, "token": live.room.players[1].token}, SimpleNamespace())
                hub.on_text(live, conn, '{"type":"ready","ready":true}')
                hub.on_frame(live, conn, b"R")
            elif transition == "round":
                hub.dispatch(live, live.room.tick(live.room.deadline))
                live.room.on_pair("RED", "CAT")
                hub.dispatch(live, live.room.tick(live.room.deadline))
            elif transition == "restart":
                live.room.phase = Phase.GAME_OVER
                hub.on_text(live, conn, '{"type":"play_again"}')
                now = time.monotonic()
                hub.dispatch(live, live.room.set_ready(1, True, now))
                hub.dispatch(live, live.room.set_ready(2, True, now))
                live.room.on_pair("RED", "CAT")
                hub.dispatch(live, live.room.tick(live.room.deadline))
            elif transition == "deadline":
                live.room.deadline = time.monotonic() - 0.001
            else:
                hub.dispatch(live, live.room.disconnect(2, time.monotonic()))
                hub.dispatch(live, live.room.reconnect(2, time.monotonic()))
            rec.release.set()
            await asyncio.sleep(0.15)
            assert live.room.players[1].index == 0
            messages = []
            while not conn.outbox.empty():
                messages.append(conn.outbox.get_nowait())
            assert not any(m and m.get("type") == "recognition" for m in messages)
        finally:
            rec.release.set()
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await hub.shutdown()
    asyncio.run(scenario())


def test_frames_are_ignored_while_paused_or_done_and_large_frames_are_rejected():
    async def scenario():
        rec = GatedRecognizer()
        rec.release.set()
        hub = Hub(FAST, lambda: rec)
        live, conn = signing(hub)
        worker = asyncio.create_task(hub.recognize(live, conn))
        try:
            hub.on_frame(live, conn, b"x" * (FAST.max_frame_bytes + 1))
            assert conn.outbox.get_nowait()["code"] == "frame_too_large"
            hub.dispatch(live, live.room.disconnect(2, time.monotonic()))
            hub.on_frame(live, conn, b"R")
            await asyncio.sleep(0.03)
            assert not rec.calls
            hub.dispatch(live, live.room.reconnect(2, time.monotonic()))
            live.room.players[1].index = len(live.room.players[1].word)
            hub.on_frame(live, conn, b"R")
            await asyncio.sleep(0.03)
            assert not rec.calls
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await hub.shutdown()
    asyncio.run(scenario())


@pytest.mark.parametrize("rejection", ["bad_frame", "frame_too_large"])
def test_rejected_image_restarts_only_the_affected_players_hold(rejection):
    class RejectingRecognizer(LetterStub):
        def process(self, jpeg):
            return None if jpeg == b"invalid" else super().process(jpeg)

    async def scenario():
        hub = Hub(replace(FAST, hold_seconds=0.2), RejectingRecognizer)
        live, conn = signing(hub)
        player, other = live.room.players[1], live.room.players[2]
        other.hold.since = time.monotonic()
        other_since = other.hold.since
        worker = asyncio.create_task(hub.recognize(live, conn))
        try:
            hub.on_frame(live, conn, b"R")
            first = await asyncio.wait_for(conn.outbox.get(), 2)
            assert first["type"] == "recognition" and not first["accepted"]
            started = player.hold.since
            data = b"invalid" if rejection == "bad_frame" else b"x" * (FAST.max_frame_bytes + 1)
            hub.on_frame(live, conn, data)
            error = await asyncio.wait_for(conn.outbox.get(), 2)
            assert error["type"] == "error" and error["code"] == rejection
            await eventually(lambda: time.monotonic() - started > 0.21)
            hub.on_frame(live, conn, b"R")
            fresh = await asyncio.wait_for(conn.outbox.get(), 2)
            assert fresh["type"] == "recognition"
            assert fresh["accepted"] is False and fresh["hold"] == 0.0
            assert player.index == 0 and player.hold.since > started
            assert other.hold.since == other_since
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await hub.shutdown()
    asyncio.run(scenario())


def test_oversized_image_discards_in_flight_and_queued_frames_before_the_reset():
    async def scenario():
        rec = GatedRecognizer()
        rec.release.set()
        hub = Hub(replace(FAST, hold_seconds=0.2), lambda: rec)
        live, conn = signing(hub)
        player = live.room.players[1]
        worker = asyncio.create_task(hub.recognize(live, conn))
        try:
            hub.on_frame(live, conn, b"R")
            assert (await asyncio.wait_for(conn.outbox.get(), 2))["accepted"] is False
            rec.started.clear()
            rec.release.clear()
            hub.on_frame(live, conn, b"R")
            await eventually(rec.started.is_set)
            hub.on_frame(live, conn, b"R")  # Queued before the rejected bytes.
            hub.on_frame(live, conn, b"x" * (FAST.max_frame_bytes + 1))
            assert conn.outbox.get_nowait()["code"] == "frame_too_large"
            rec.release.set()
            await asyncio.sleep(0.25)  # Let both old CPU work and the throttle finish.
            assert conn.outbox.empty(), "pre-rejection frames must not restart or accept a hold"
            assert player.hold.since is None and player.index == 0
            hub.on_frame(live, conn, b"R")
            fresh = await asyncio.wait_for(conn.outbox.get(), 2)
            assert fresh["accepted"] is False and fresh["hold"] == 0.0
        finally:
            rec.release.set()
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await hub.shutdown()
    asyncio.run(scenario())


def test_expired_room_cannot_be_revived_between_ticks():
    async def scenario():
        hub = Hub(replace(FAST, reconnect_grace=0), LetterStub)
        live, conn, _ = hub.join({"type": "create_room"}, SimpleNamespace())
        hub.dispatch(live, live.room.disconnect(1, time.monotonic()))
        reply = hub.join({"type": "rejoin", "room": live.room.code, "token": live.room.players[1].token}, SimpleNamespace())
        assert reply["code"] == "room_not_found"
        assert not hub.rooms
        await hub.shutdown()
    asyncio.run(scenario())


def test_disconnect_and_shutdown_await_recognizer_cleanup():
    instances = []

    def factory():
        rec = GatedRecognizer()
        rec.release.set()
        instances.append(rec)
        return rec

    app = create_app(FAST, make_recognizer=factory)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "create_room"})
            recv(ws, "room_joined")
        # The disconnect handler has finished before the TestClient session exits.
    assert instances and all(rec.closed.is_set() for rec in instances)
    assert not app.state.hub.rooms


@pytest.mark.parametrize("stage", ["construction", "processing"])
def test_shutdown_waits_for_already_cancelled_native_work(stage):
    async def scenario():
        rec = GatedRecognizer()
        constructing = threading.Event()
        allow_construction = threading.Event()

        def factory():
            constructing.set()
            assert allow_construction.wait(2)
            return rec

        hub = Hub(FAST, factory)
        live, conn = signing(hub)
        conn.worker = hub.spawn(live, hub.recognize(live, conn))
        await eventually(constructing.is_set)
        if stage == "processing":
            allow_construction.set()
            hub.on_frame(live, conn, b"R")
            await eventually(rec.started.is_set)
        conn.worker.cancel()
        await asyncio.sleep(0)
        shutdown = asyncio.create_task(hub.shutdown())
        try:
            await asyncio.sleep(0.02)
            assert not shutdown.done(), "shutdown must await CPU work and close"
        finally:
            allow_construction.set()
            rec.release.set()
            await shutdown
        assert rec.closed.is_set()
        assert not live.tasks
    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["words", "image"])
def test_muse_timeout_falls_back_and_does_not_leave_tasks(kind, monkeypatch):
    async def scenario():
        hub = Hub(replace(FAST, word_timeout=0.02, image_timeout=0.02), LetterStub)
        hub.muse = object()
        live, conn, _ = hub.join({"type": "create_room"}, SimpleNamespace())

        async def slow(*args):
            await asyncio.sleep(10)

        if kind == "words":
            monkeypatch.setattr("server.app.muse.fetch_pair", slow)
            await asyncio.wait_for(hub.fetch_pair(live, frozenset(), 0, 1), 0.5)
            assert live.room.next_pair is None
            hub.join({"type": "join_room", "room": live.room.code}, SimpleNamespace())
            now = time.monotonic()
            hub.dispatch(live, live.room.set_ready(1, True, now))
            hub.dispatch(live, live.room.set_ready(2, True, now))
            hub.dispatch(live, live.room.tick(live.room.deadline))
            assert live.room.players[1].word and live.room.players[2].word
        else:
            from server.game import GenerateImage

            monkeypatch.setattr("server.app.muse.generate_image", slow)
            live.room.phase = Phase.GENERATING
            live.room.round_no = 1
            live.room.phrase = "RED CAT"
            await asyncio.wait_for(hub.generate(live, GenerateImage(1, "RED CAT"), 0), 0.5)
            assert live.room.score == 100
            assert live.images["1"][1] == "image/svg+xml"
            assert live.room.phase is Phase.REVEAL  # the placeholder shows, then the game moves on by itself
            live.room.tick(live.room.deadline)
            assert live.room.phase is Phase.COUNTDOWN
        await hub.shutdown()
        assert not live.tasks
    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["words", "image"])
def test_muse_result_from_a_previous_game_is_discarded(kind, monkeypatch):
    async def scenario():
        from server.game import GenerateImage

        hub = Hub(FAST, LetterStub)
        hub.muse = object()
        live, conn, _ = hub.join({"type": "create_room"}, SimpleNamespace())
        started, release = asyncio.Event(), asyncio.Event()

        async def delayed(*args):
            started.set()
            await release.wait()
            return ("RED", "CAT") if kind == "words" else (b"jpeg", "image/jpeg")

        if kind == "words":
            monkeypatch.setattr("server.app.muse.fetch_pair", delayed)
            task = hub.spawn(live, hub.fetch_pair(live, frozenset(), 0, 1))
        else:
            monkeypatch.setattr("server.app.muse.generate_image", delayed)
            live.room.phase, live.room.round_no, live.room.phrase = Phase.GENERATING, 1, "RED CAT"
            task = hub.spawn(live, hub.generate(live, GenerateImage(1, "RED CAT"), 0))
        await started.wait()
        live.room.phase = Phase.GAME_OVER
        hub.on_text(live, conn, '{"type":"play_again"}')
        # A new game's first generation has the same round number as the old game.
        if kind == "image":
            live.room.phase, live.room.round_no, live.room.phrase = Phase.GENERATING, 1, "BIG DOG"
        release.set()
        await task
        assert live.room.next_pair is None
        assert live.room.score == 0 and not live.images
        await hub.shutdown()
    asyncio.run(scenario())
