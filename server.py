"""Serve the SignSync site and generate shared images from two words.

Run:
    python server.py
Then open http://localhost:8000
"""

import json
import re
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

from image_generation import generate_image

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "generated"
WORD_PATTERN = re.compile(r"^[A-Za-z][A-Za-z '-]{0,29}$")
PORT = 8000


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _is_blocked(self, path: str) -> bool:
        """Never serve dotfiles/folders (.env, .git, .venv) or Python source."""
        parts = [part for part in path.split("/") if part]
        return any(part.startswith(".") for part in parts) or path.endswith(".py")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/generate":
            self._handle_generate(parse_qs(url.query))
            return
        if self._is_blocked(url.path):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        super().do_GET()

    def _handle_generate(self, query: dict) -> None:
        word1 = query.get("word1", [""])[0].strip()
        word2 = query.get("word2", [""])[0].strip()

        if not WORD_PATTERN.match(word1) or not WORD_PATTERN.match(word2):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Provide two valid words."})
            return

        OUTPUT_DIR.mkdir(exist_ok=True)
        output = OUTPUT_DIR / f"{uuid.uuid4().hex}.png"

        try:
            generate_image(word1, word2, str(output))
        except (requests.RequestException, ValueError, RuntimeError) as error:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
            return

        self._send_json(
            HTTPStatus.OK,
            {"image": f"/generated/{output.name}", "word1": word1, "word2": word2},
        )


if __name__ == "__main__":
    print(f"Serving on http://localhost:{PORT}")
    ThreadingHTTPServer(("localhost", PORT), Handler).serve_forever()
