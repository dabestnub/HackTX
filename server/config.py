import os
from dataclasses import dataclass, field
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"


@dataclass(frozen=True)
class Settings:
    """Game and server tunables. Tests pass faster values; production uses the defaults."""

    time_limit: float = 10.0       # seconds per round for the whole word (Q13)
    hold_seconds: float = 0.6      # Q14
    min_confidence: float = 0.8    # Q14
    countdown: float = 3.0
    reveal: float = 5.0            # seconds an image shows before the next countdown
    reconnect_grace: float = 30.0  # Q8
    points_per_round: int = 100    # Q3
    target_score: int = 500        # Q3
    image_timeout: float = 10.0    # Q19
    word_timeout: float = 8.0
    max_frame_bytes: int = 300_000
    max_frame_width: int = 640     # Q16
    max_frame_height: int = 480
    api_key: str | None = field(default_factory=lambda: os.environ.get("MODEL_API_KEY") or None)
    muse_base_url: str = "https://api.meta.ai/v1"
