import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from image_generation import build_combined_phrase, build_prompt, generate_image


class ImageGenerationTests(unittest.TestCase):
    def test_build_combined_phrase(self):
        self.assertEqual(build_combined_phrase("  moon", "river  "), "moon river")

    def test_build_prompt_mentions_combined_phrase(self):
        prompt = build_prompt("moon", "river")
        self.assertIn("moon river", prompt)
        self.assertIn("moon", prompt)
        self.assertIn("river", prompt)

    def test_generate_image_rejects_empty_words(self):
        with self.assertRaises(ValueError):
            generate_image("", "flower", "out.png")

    def test_generate_image_requires_png_output(self):
        with self.assertRaises(ValueError):
            generate_image("sun", "flower", "out.jpg")

    def test_generate_image_requires_api_key(self):
        old_key = os.environ.pop("META_MUSE_API_KEY", None)
        self.addCleanup(lambda: os.environ.__setitem__("META_MUSE_API_KEY", old_key) if old_key else os.environ.pop("META_MUSE_API_KEY", None))
        with self.assertRaises(ValueError):
            generate_image("sun", "flower", "out.png")

    @patch("image_generation.requests.post")
    def test_generate_image_success(self, mock_post):
        os.environ["META_MUSE_API_KEY"] = "fake-key"
        os.environ["META_MUSE_MODEL"] = "muse-image-1.0"
        os.environ["META_MUSE_SIZE"] = "1024x1024"

        raw = b"fake-bytes"
        mock_post.return_value.ok = True
        mock_post.return_value.json.return_value = {
            "data": [{"b64_json": base64.b64encode(raw).decode("utf-8")}]
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "result.png"
            saved_path = generate_image("sun", "flower", str(out))
            self.assertTrue(saved_path.exists())
            self.assertEqual(saved_path.read_bytes(), raw)

        call_kwargs = mock_post.call_args.kwargs
        self.assertIn("sun flower", call_kwargs["json"]["prompt"])
        self.assertEqual(call_kwargs["json"]["model"], "muse-image-1.0")
        self.assertEqual(call_kwargs["json"]["size"], "1024x1024")


if __name__ == "__main__":
    unittest.main()