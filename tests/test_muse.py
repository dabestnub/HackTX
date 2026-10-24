import asyncio
import base64
import json

import httpx

from server import muse

BASE = "https://api.meta.ai/v1"


def call(handler, fn, *args):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE) as client:
            return await fn(client, *args)
    return asyncio.run(go())


def test_fetch_pair_sends_structured_output_request_and_parses_it():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        content = json.dumps({"adjective": "red", "noun": "cat"})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    assert call(handler, muse.fetch_pair, {"BIG"}) == ("red", "cat")
    assert seen["url"] == f"{BASE}/chat/completions"
    assert seen["body"]["model"] == "muse-spark-1.3"
    fmt = seen["body"]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False
    assert "BIG" in seen["body"]["messages"][0]["content"]


def test_fetch_pair_returns_none_on_errors():
    assert call(lambda r: httpx.Response(500), muse.fetch_pair, set()) is None
    assert call(lambda r: httpx.Response(200, json={"choices": []}), muse.fetch_pair, set()) is None


def test_generate_image_uses_low_reasoning_without_search():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(b"img").decode()}]})

    assert call(handler, muse.generate_image, "RED CAT") == (b"img", "image/jpeg")
    assert seen["url"] == f"{BASE}/images/generations"
    body = seen["body"]
    assert body["model"] == "muse-image-1.0"
    assert body["reasoning_strength"] == "low"
    assert body["tool_enablement"] == {"enable_web_search": False, "enable_image_search": False}
    assert body["response_format"] == "b64_json" and body["output_format"] == "jpeg"
    assert "red cat" in body["prompt"]


def test_generate_image_returns_none_on_errors():
    assert call(lambda r: httpx.Response(400, json={"error": "bad"}), muse.generate_image, "RED CAT") is None


def test_placeholder_shows_the_phrase():
    svg = muse.placeholder_svg("RED CAT").decode()
    assert svg.startswith("<svg") and "RED CAT" in svg
    assert "&lt;b&gt;" in muse.placeholder_svg("<b>").decode()
