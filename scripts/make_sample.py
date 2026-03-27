"""Synthesize a spoken question to test /chat without recording anything.

    python scripts/make_sample.py "how much is the gigabit plan"

Writes 16 kHz mono WAV -- the same format a browser sends -- to data/samples/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.audio import resample, write_wav  # noqa: E402
from app.config import settings  # noqa: E402
from app.tts import TTS  # noqa: E402


def main() -> int:
    text = " ".join(sys.argv[1:]) or "How much does the gigabit plan cost per month?"
    out = Path(sys.argv[0]).parent.parent / "data" / "samples" / "question.wav"

    tts = TTS()
    pcm = tts.synthesize(text).astype(np.float32) / 32767.0
    write_wav(out, resample(pcm, tts.sample_rate, settings.sample_rate), settings.sample_rate)

    print(f'Wrote {out} ({len(pcm) / tts.sample_rate:.1f}s): "{text}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
