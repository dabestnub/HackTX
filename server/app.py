"""HTTP and WebSocket I/O around the pure room engine."""
import asyncio
import json
import logging
import secrets
import string
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from anyio import CancelScope
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from . import muse
from .config import MODELS, Settings
from .game import CloseRoom, FetchPair, GenerateImage, Phase, Room, Send

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"
TICK = 0.1
FRAME_GAP = 0.1


def error_msg(code: str, message: str) -> dict:
    return {"type": "error", "code": code, "message": message}


@dataclass(eq=False)
class Conn:
    slot: int
    ws: WebSocket
    outbox: asyncio.Queue = field(default_factory=asyncio.Queue)
    frame: tuple[bytes, int] | None = None
    frame_ready: asyncio.Event = field(default_factory=asyncio.Event)
    frame_epoch: int = 0
    worker: asyncio.Task | None = None
    sender: asyncio.Task | None = None


@dataclass(eq=False)
class Live:
    room: Room
    conns: dict[int, Conn] = field(default_factory=dict)
    images: dict[str, tuple[bytes, str]] = field(default_factory=dict)
    tasks: set[asyncio.Task] = field(default_factory=set)
    epoch: int = 0
    state_key: tuple | None = None
    game_id: int = 0


class Hub:
    """Own sockets, worker lifetimes, timers, and Muse calls for in-memory rooms."""

    def __init__(self, settings: Settings, make_recognizer):
        self.s = settings
        self.make_recognizer = make_recognizer
        self.rooms: dict[str, Live] = {}
        self.muse = None
        self.closing: set[asyncio.Task] = set()

    def spawn(self, live: Live, coro):
        task = asyncio.create_task(coro)
        live.tasks.add(task)
        task.add_done_callback(live.tasks.discard)
        return task

    def sync_frames(self, live: Live):
        room = live.room
        key = (room.phase, room.round_no, room.deadline, room.paused_at)
        if key != live.state_key:
            live.state_key = key
            live.epoch += 1
            for conn in live.conns.values():
                conn.frame = None
                conn.frame_ready.clear()

    def dispatch(self, live: Live, outs: list):
        self.sync_frames(live)
        for out in outs:
            if isinstance(out, Send):
                for slot, conn in live.conns.items():
                    if out.to in (None, slot):
                        conn.outbox.put_nowait(out.msg)
            elif isinstance(out, FetchPair):
                self.spawn(live, self.fetch_pair(live, out.used, live.game_id, live.room.round_no + 1))
            elif isinstance(out, GenerateImage):
                self.spawn(live, self.generate(live, out, live.game_id))
            elif isinstance(out, CloseRoom):
                self.close(live)

    def close(self, live: Live):
        if self.rooms.pop(live.room.code, None) is None:
            return
        live.room.closed = True
        senders = {conn.sender for conn in live.conns.values() if conn.sender is not None}
        for conn in live.conns.values():
            conn.frame = None
            conn.frame_ready.clear()
            conn.outbox.put_nowait(None)
        current = asyncio.current_task()
        tasks = list(live.tasks)
        for task in tasks:
            if task not in senders and task is not current and not task.cancelling():
                task.cancel()
        cleanup = asyncio.create_task(self.finish_close(tasks))
        self.closing.add(cleanup)
        cleanup.add_done_callback(self.closing.discard)

    async def finish_close(self, tasks):
        await asyncio.gather(*tasks, return_exceptions=True)

    async def shutdown(self):
        for live in list(self.rooms.values()):
            self.close(live)
        await asyncio.gather(*self.closing, return_exceptions=True)

    def new_code(self):
        while True:
            code = "".join(secrets.choice(string.ascii_uppercase) for _ in range(4))
            if code not in self.rooms:
                return code

    async def ticker(self, live: Live):
        while not live.room.closed:
            await asyncio.sleep(TICK)
            self.dispatch(live, live.room.tick(time.monotonic()))

    async def fetch_pair(self, live: Live, used, game_id: int, target_round: int):
        if self.muse is None:
            return
        try:
            pair = await asyncio.wait_for(muse.fetch_pair(self.muse, used), self.s.word_timeout)
        except TimeoutError:
            pair = None
        if pair and not live.room.closed and live.game_id == game_id and live.room.round_no < target_round:
            self.dispatch(live, live.room.on_pair(*pair))

    async def generate(self, live: Live, job: GenerateImage, game_id: int):
        image = None
        if self.muse is not None:
            try:
                image = await asyncio.wait_for(muse.generate_image(self.muse, job.phrase), self.s.image_timeout)
            except TimeoutError:
                log.warning("Image generation exceeded %s seconds", self.s.image_timeout)
        if (live.room.closed or live.game_id != game_id or live.room.round_no != job.round_no
                or live.room.phase is not Phase.GENERATING):
            return
        image = image or (muse.placeholder_svg(job.phrase), "image/svg+xml")
        image_id = str(len(live.images) + 1)
        live.images[image_id] = image
        self.dispatch(live, live.room.on_image(job.round_no, f"/images/{live.room.code}/{image_id}", time.monotonic()))

    async def sender(self, conn: Conn):
        try:
            while (msg := await conn.outbox.get()) is not None:
                await conn.ws.send_json(msg)
            await conn.ws.close()
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass

    def can_recognize(self, live: Live, conn: Conn):
        room = live.room
        player = room.players.get(conn.slot)
        return (not room.closed and live.conns.get(conn.slot) is conn and room.phase is Phase.SIGNING
                and room.paused_at is None and player is not None and not player.done
                and room.deadline is not None and time.monotonic() < room.deadline)

    async def recognize(self, live: Live, conn: Conn):
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(max_workers=1)
        rec = None
        construction = loop.run_in_executor(pool, self.make_recognizer)
        try:
            rec = await asyncio.shield(construction)
            while True:
                await conn.frame_ready.wait()
                conn.frame_ready.clear()
                pending, conn.frame = conn.frame, None
                if pending is None or pending[1] != live.epoch or not self.can_recognize(live, conn):
                    continue
                jpeg, epoch = pending
                frame_epoch = conn.frame_epoch
                started = time.monotonic()
                result = await asyncio.shield(loop.run_in_executor(pool, rec.process, jpeg))
                if epoch == live.epoch and frame_epoch == conn.frame_epoch and self.can_recognize(live, conn):
                    if result is None:
                        live.room.players[conn.slot].hold.reset()
                        conn.outbox.put_nowait(error_msg("bad_frame", "Frames must be JPEG images of at most 640x480."))
                    else:
                        self.dispatch(live, live.room.on_recognition(conn.slot, result.probs, result.landmarks, time.monotonic()))
                await asyncio.sleep(max(0.0, FRAME_GAP - (time.monotonic() - started)))
        except Exception:
            log.exception("Recognition worker failed")
            if live.conns.get(conn.slot) is conn:
                conn.outbox.put_nowait(error_msg("recognizer_unavailable", "Camera recognition is unavailable."))
                conn.outbox.put_nowait(None)
        finally:
            try:
                # Shield construction so cancellation still retrieves and closes the native resource.
                if rec is None:
                    rec = await asyncio.shield(construction)
                await loop.run_in_executor(pool, rec.close)
            finally:
                pool.shutdown(wait=False)

    def join(self, message: dict, ws: WebSocket):
        now = time.monotonic()
        kind = message.get("type")
        name = str(message.get("name") or "Player")[:20]
        if kind == "create_room":
            live = Live(Room(self.new_code(), self.s))
            self.rooms[live.room.code] = live
            self.spawn(live, self.ticker(live))
            slot, outs = live.room.join(name, secrets.token_urlsafe(16), now)
        elif kind in ("join_room", "rejoin"):
            live = self.rooms.get(str(message.get("room", "")).strip().upper())
            if live is not None:
                self.dispatch(live, live.room.tick(now))
            if live is None or live.room.closed:
                return error_msg("room_not_found", "No room with that code.")
            if kind == "join_room":
                if live.room.full:
                    return error_msg("room_full", "That room already has two players.")
                slot, outs = live.room.join(name, secrets.token_urlsafe(16), now)
            else:
                slot = live.room.slot_for_token(str(message.get("token", "")))
                if slot is None:
                    return error_msg("bad_token", "That player token is not valid for this room.")
                old = live.conns.get(slot)
                if old is not None:
                    old.frame = None
                    old.frame_ready.clear()
                    old.outbox.put_nowait(None)
                    if old.worker is not None and not old.worker.cancelling():
                        old.worker.cancel()
                outs = live.room.reconnect(slot, now)
        else:
            return error_msg("join_first", "Send create_room, join_room or rejoin first.")
        conn = Conn(slot, ws)
        live.conns[slot] = conn
        return live, conn, outs

    def on_frame(self, live: Live, conn: Conn, data: bytes):
        if not self.can_recognize(live, conn):
            return
        if len(data) > self.s.max_frame_bytes:
            live.room.players[conn.slot].hold.reset()
            conn.frame_epoch += 1  # Invalidate CPU work that predates this rejected frame.
            conn.frame = None
            conn.frame_ready.clear()
            conn.outbox.put_nowait(error_msg("frame_too_large", f"Frames must be under {self.s.max_frame_bytes} bytes."))
            return
        conn.frame = (data, live.epoch)
        conn.frame_ready.set()

    def on_text(self, live: Live, conn: Conn, text: str):
        if live.room.closed or live.conns.get(conn.slot) is not conn:
            return
        try:
            message = json.loads(text)
        except ValueError:
            message = None
        if not isinstance(message, dict):
            conn.outbox.put_nowait(error_msg("bad_message", "Messages must be JSON objects."))
            return
        now = time.monotonic()
        kind = message.get("type")
        if kind == "ready":
            self.dispatch(live, live.room.set_ready(conn.slot, bool(message.get("ready", True)), now))
        elif kind == "play_again":
            if live.room.phase is Phase.GAME_OVER:
                live.game_id += 1
            self.dispatch(live, live.room.play_again(conn.slot, now))
        else:
            conn.outbox.put_nowait(error_msg("unknown_type", f"Unknown message type {kind!r}."))

    async def serve(self, ws: WebSocket):
        await ws.accept()
        first = await ws.receive()
        if first["type"] == "websocket.disconnect":
            return
        try:
            message = json.loads(first.get("text") or "")
        except ValueError:
            message = None
        joined = self.join(message, ws) if isinstance(message, dict) else error_msg("join_first", "Send create_room, join_room or rejoin first.")
        if isinstance(joined, dict):
            await ws.send_json(joined)
            await ws.close()
            return
        live, conn, outs = joined
        conn.sender = self.spawn(live, self.sender(conn))
        conn.worker = self.spawn(live, self.recognize(live, conn))
        self.dispatch(live, outs)
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    self.on_frame(live, conn, message["bytes"])
                elif message.get("text") is not None:
                    self.on_text(live, conn, message["text"])
        except WebSocketDisconnect:
            pass
        finally:
            if live.conns.get(conn.slot) is conn and not live.room.closed:
                del live.conns[conn.slot]
                self.dispatch(live, live.room.disconnect(conn.slot, time.monotonic()))
            if not conn.worker.cancelling():
                conn.worker.cancel()
            conn.sender.cancel()
            # ASGI disconnect cancellation must not interrupt native-resource cleanup.
            with CancelScope(shield=True):
                await asyncio.gather(conn.worker, conn.sender, return_exceptions=True)


def create_app(settings: Settings | None = None, make_recognizer=None) -> FastAPI:
    settings = settings or Settings()
    if make_recognizer is None:
        from .classifier import Classifier
        from .recognizer import Recognizer

        classifier = Classifier(MODELS / "classifier.npz")

        def make_recognizer():
            return Recognizer(classifier, max_w=settings.max_frame_width, max_h=settings.max_frame_height)

    hub = Hub(settings, make_recognizer)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.api_key:
            hub.muse = muse.make_client(settings.api_key, settings.muse_base_url, settings.image_timeout)
        try:
            yield
        finally:
            await hub.shutdown()
            if hub.muse is not None:
                await hub.muse.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.hub = hub
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "rooms": len(hub.rooms)}

    @app.get("/images/{code}/{image_id}")
    def image(code: str, image_id: str):
        live = hub.rooms.get(code)
        found = live.images.get(image_id) if live else None
        return Response(found[0], media_type=found[1]) if found else Response(status_code=404)

    @app.get("/dev")
    def dev():
        return FileResponse(STATIC / "dev.html")

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        await hub.serve(ws)

    return app
