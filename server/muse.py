"""Muse (Meta Model API) calls. Each returns None on failure so the caller falls back (Q7, Q19)."""
import base64
import json
import logging
from html import escape

import httpx

log = logging.getLogger(__name__)

TEXT_MODEL = "muse-spark-1.3"  # not a -contributor variant, so Meta does not train on our prompts
IMAGE_MODEL = "muse-image-1.0"

PAIR_SCHEMA = {
    "name": "word_pair",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"adjective": {"type": "string"}, "noun": {"type": "string"}},
        "required": ["adjective", "noun"],
        "additionalProperties": False,
    },
}


def make_client(api_key: str, base_url: str, timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)


async def fetch_pair(client: httpx.AsyncClient, used) -> tuple[str, str] | None:
    avoid = ", ".join(sorted(used)) or "none"
    prompt = (
        "Pick one common English adjective and one common concrete noun that together make a fun, "
        "easy-to-draw phrase, like RED CAT. Each word must be 3 to 5 letters long, use only the letters A-Z, "
        f"and must not contain the letters J or Z. Do not use any of these words: {avoid}."
    )
    try:
        r = await client.post("/chat/completions", json={
            "model": TEXT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": PAIR_SCHEMA},
        })
        r.raise_for_status()
        data = json.loads(r.json()["choices"][0]["message"]["content"])
        return data["adjective"], data["noun"]
    except Exception:  # any failure falls back to the built-in words
        log.exception("Muse word pair failed")
        return None


async def generate_image(client: httpx.AsyncClient, phrase: str) -> tuple[bytes, str] | None:
    try:
        r = await client.post("/images/generations", json={
            "model": IMAGE_MODEL,
            "prompt": f"A bright, friendly cartoon illustration of a {phrase.lower()}, simple background, no text.",
            "n": 1,
            "size": "1024x1024",
            "response_format": "b64_json",
            "output_format": "jpeg",
            "reasoning_strength": "low",  # faster (Q19)
            "tool_enablement": {"enable_web_search": False, "enable_image_search": False},
        })
        r.raise_for_status()
        return base64.b64decode(r.json()["data"][0]["b64_json"]), "image/jpeg"
    except Exception:  # any failure shows the placeholder instead
        log.exception("Muse image failed")
        return None


def placeholder_svg(phrase: str) -> bytes:
    """Shown when Muse is off, fails, or takes longer than the image timeout (Q19)."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">'
        '<rect width="1024" height="1024" fill="#f4efe6"/>'
        '<text x="512" y="540" font-family="sans-serif" font-size="96" font-weight="700" '
        f'text-anchor="middle" fill="#333">{escape(phrase)}</text></svg>'
    ).encode()
