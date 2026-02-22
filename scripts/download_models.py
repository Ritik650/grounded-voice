"""Fetch the model files that are too large to keep in git.

Run once after cloning, and again in the Docker build so the first request of a fresh
container doesn't pay for a download on top of warmup.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

SILERO_VAD = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ok      {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetch   {dest.relative_to(ROOT)} <- {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed, hard-coded hosts
    tmp.replace(dest)


def main() -> int:
    voice = sys.argv[1] if len(sys.argv) > 1 else "en_US-lessac-medium"
    print("Downloading models...")
    fetch(SILERO_VAD, MODELS / "silero_vad.onnx")
    fetch(f"{PIPER_BASE}/{voice}.onnx", MODELS / "piper" / f"{voice}.onnx")
    fetch(f"{PIPER_BASE}/{voice}.onnx.json", MODELS / "piper" / f"{voice}.onnx.json")

    # Pull the embedding + BM25 models into the fastembed cache now, for the same
    # reason: a cold container should not download anything on its first request.
    try:
        from fastembed import SparseTextEmbedding, TextEmbedding

        from app.config import settings

        print("  cache   embedding models")
        TextEmbedding(settings.embed_model)
        SparseTextEmbedding("Qdrant/bm25")
    except ImportError:
        print("  skip    embedding cache (fastembed not installed yet)")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
