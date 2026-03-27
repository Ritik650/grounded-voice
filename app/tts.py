"""Text to speech: Piper (VITS, ONNX Runtime).

Measured on this machine, Piper's first synthesis costs ~7.7 s and every subsequent
one runs at ~0.12x real time. That gap is ONNX Runtime's graph optimisation on first
inference, so `warmup()` is called at server start -- otherwise the first user of a
fresh process eats the whole cost and concludes the assistant is broken.

Output stays at the voice's native 22.05 kHz rather than being downsampled to the
16 kHz the input side uses. Speech intelligibility lives in exactly the band that
downsampling would remove, and the browser's AudioContext resamples for free on
playback, so the only cost is carrying a sample rate on the wire.
"""

from __future__ import annotations

import logging
import math
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import settings

log = logging.getLogger(__name__)


@dataclass
class AudioFrame:
    pcm: bytes  # signed 16-bit little-endian mono
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate


class PiperEngine:
    name = "piper"

    def __init__(self, model_path: Path | None = None):
        from piper import PiperVoice, SynthesisConfig

        path = Path(model_path or settings.piper_model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Piper voice missing at {path}. Run: python scripts/download_models.py"
            )
        self.voice = PiperVoice.load(path)
        self.sample_rate = self.voice.config.sample_rate
        self._syn_config = SynthesisConfig(
            length_scale=settings.tts_length_scale, normalize_audio=True
        )

    def synthesize(self, text: str) -> np.ndarray:
        chunks = list(self.voice.synthesize(text, syn_config=self._syn_config))
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate([c.audio_int16_array for c in chunks])

    def stream(self, text: str, frame_ms: int | None = None) -> Iterator[AudioFrame]:
        """Yield fixed-size frames as Piper produces them.

        Slicing into small frames is what makes barge-in feel instant: the consumer
        can stop pulling between frames, so the worst-case delay between the user
        interrupting and the audio stopping is one frame, not one sentence.
        """
        frame_bytes = _frame_bytes(frame_ms or settings.tts_chunk_ms, self.sample_rate)
        tail = b""
        for chunk in self.voice.synthesize(text, syn_config=self._syn_config):
            buf = tail + chunk.audio_int16_bytes
            n_full = len(buf) // frame_bytes
            for i in range(n_full):
                yield AudioFrame(buf[i * frame_bytes : (i + 1) * frame_bytes], self.sample_rate)
            tail = buf[n_full * frame_bytes :]
        if tail:
            yield AudioFrame(tail, self.sample_rate)


class NullEngine:
    """Test-only engine: a quiet tone whose length tracks the text.

    Lets CI exercise the full pipeline -- frame sizes, wire protocol, barge-in
    cancellation -- without downloading a 63 MB voice model. It is not a fallback
    voice; nobody should ever hear it.
    """

    name = "null"

    def __init__(self, sample_rate: int = 22_050):
        self.sample_rate = sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        duration = max(0.2, len(text.split()) / 3.0)  # ~180 wpm
        t = np.arange(int(duration * self.sample_rate)) / self.sample_rate
        return (0.05 * np.sin(2 * math.pi * 220 * t) * 32767).astype(np.int16)

    def stream(self, text: str, frame_ms: int | None = None) -> Iterator[AudioFrame]:
        frame_bytes = _frame_bytes(frame_ms or settings.tts_chunk_ms, self.sample_rate)
        pcm = self.synthesize(text).tobytes()
        for i in range(0, len(pcm), frame_bytes):
            yield AudioFrame(pcm[i : i + frame_bytes], self.sample_rate)


class TTS:
    """Engine-agnostic facade used by the pipeline and the WebSocket handler."""

    def __init__(self, engine: str | None = None):
        self.engine = _build_engine(engine or settings.tts_engine)
        self.sample_rate = self.engine.sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        return self.engine.synthesize(text)

    def stream(self, text: str, frame_ms: int | None = None) -> Iterator[AudioFrame]:
        return self.engine.stream(text, frame_ms=frame_ms)

    def to_wav(self, text: str, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(self.synthesize(text).tobytes())
        return path

    def warmup(self) -> float:
        """Pay the first-inference cost at startup. Returns seconds spent."""
        import time

        start = time.perf_counter()
        self.synthesize("Ready.")
        elapsed = time.perf_counter() - start
        log.info("TTS warmup (%s) took %.2fs", self.engine.name, elapsed)
        return elapsed


def _build_engine(name: str):
    if name == "auto":
        name = "piper" if Path(settings.piper_model_path).exists() else "null"
        if name == "null":
            log.warning("No Piper voice found; using the silent test engine.")
    if name == "piper":
        return PiperEngine()
    if name == "null":
        return NullEngine()
    raise ValueError(f"Unknown TTS engine {name!r}. Options: auto, piper, null")


def _frame_bytes(frame_ms: int, sample_rate: int) -> int:
    # Even byte count: an odd split would tear a 16-bit sample across two frames.
    return max(2, (frame_ms * sample_rate // 1000) * 2)
