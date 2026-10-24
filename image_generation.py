"""Generate an image combining two words.

Example:
    python3 image_generation.py "sunset" "dragon" --output sunset_dragon.png
"""

import argparse
import base64
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# Load .env from the same folder as this Python file.
load_dotenv(Path(__file__).resolve().parent / ".env")

ENDPOINT = "https://api.meta.ai/v1/images/generations"


def build_combined_phrase(word1: str, word2: str) -> str:
    """Normalize and combine two words into a single phrase."""
    first = word1.strip()
    second = word2.strip()
    if not first or not second:
        raise ValueError("Both words must be non-empty.")
    return f"{first} {second}".strip()


def build_prompt(word1: str, word2: str) -> str:
    """Create a prompt that fuses two words into one image concept."""
    combined_phrase = build_combined_phrase(word1, word2)
    return (
        "Create one playful cartoon illustration of a single fused concept: "
        f"'{combined_phrase}'. "
        f"Clearly include elements of '{word1.strip()}' and '{word2.strip()}' "
        "within one cohesive scene. Use cheerful colors and a simple background. "
        "No text, letters, captions, or watermarks."
    )


def generate_image(word1: str, word2: str, output: str) -> Path:
    """Generate an image from two words and save it as a PNG."""
    word1 = word1.strip()
    word2 = word2.strip()
    combined_phrase = build_combined_phrase(word1, word2)

    api_key = os.getenv("META_MUSE_API_KEY")
    if not api_key:
        raise ValueError("Add META_MUSE_API_KEY to your .env file.")

    output_path = Path(output)

    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")
    elif output_path.suffix.lower() != ".png":
        raise ValueError("The output filename must end in .png.")

    prompt = build_prompt(word1, word2)

    print(f"Generating an image for combined concept: '{combined_phrase}'...")

    endpoint = os.getenv("META_MUSE_ENDPOINT", ENDPOINT)
    model = os.getenv("META_MUSE_MODEL", "muse-image-1.0")
    size = os.getenv("META_MUSE_SIZE", "1024x1024")

    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json",
            "output_format": "png",
        },
        timeout=180,
    )

    if not response.ok:
        raise RuntimeError(
            f"Meta API error {response.status_code}: {response.text}"
        )

    result = response.json()
    images = result.get("data")

    if not images or not images[0].get("b64_json"):
        raise RuntimeError("The API did not return image data.")

    image_bytes = base64.b64decode(
        images[0]["b64_json"],
        validate=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)

    return output_path.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Generate an image combining two words."
    )
    parser.add_argument("word1", help="First word")
    parser.add_argument("word2", help="Second word")
    parser.add_argument(
        "--output",
        default="generated_image.png",
        help="Where to save the PNG image",
    )
    args = parser.parse_args()

    try:
        image_path = generate_image(
            args.word1,
            args.word2,
            args.output,
        )
    except (requests.RequestException, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Image saved to: {image_path}")


if __name__ == "__main__":
    main()