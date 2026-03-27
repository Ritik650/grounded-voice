"""Audio plumbing shared by every stage.

The whole pipeline speaks one internal format: float32 mono at `settings.sample_rate`,
in [-1, 1]. Conversion happens exactly twice -- once at the wire boundary (PCM16 in
from the browser) and once on the way out (PCM16 back). Keeping a single internal
representation is what stops resampling bugs from hiding between stages.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .config import settings

INT16_MAX = 32767.0


def pcm16_to_float32(raw: bytes) -> np.ndarray:
    """Little-endian signed 16-bit PCM -> float32 in [-1, 1]."""
    if len(raw) % 2:  # a frame split mid-sample; drop the stray byte
        raw = raw[:-1]
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / INT16_MAX


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """float32 in [-1, 1] -> little-endian signed 16-bit PCM."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * INT16_MAX).astype("<i2").tobytes()


def resample(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return samples.astype(np.float32, copy=False)
    return soxr.resample(samples, src_sr, dst_sr, quality="HQ").astype(np.float32)


def load_audio(path: str | Path, target_sr: int | None = None) -> np.ndarray:
    """Read any soundfile-supported container to mono float32 at `target_sr`.

    Falls back to PyAV (bundled with faster-whisper, no system ffmpeg needed) for
    compressed formats soundfile can't open -- notably the webm/opus that browsers
    produce by default.
    """
    target_sr = target_sr or settings.sample_rate
    try:
        samples, src_sr = sf.read(str(path), dtype="float32", always_2d=True)
        samples = samples.mean(axis=1)  # downmix to mono
    except Exception:
        samples, src_sr = _decode_with_av(path, target_sr)
    return resample(samples, src_sr, target_sr)


def _decode_with_av(path: str | Path, target_sr: int) -> tuple[np.ndarray, int]:
    from faster_whisper.audio import decode_audio  # lazy: only needed for odd formats

    return decode_audio(str(path), sampling_rate=target_sr), target_sr


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate or settings.sample_rate, subtype="PCM_16")
    return path


def wav_bytes(samples: np.ndarray, sample_rate: int | None = None) -> bytes:
    """In-memory WAV, for returning synthesized audio over HTTP without a temp file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate or settings.sample_rate)
        w.writeframes(float32_to_pcm16(samples))
    return buf.getvalue()


def duration_s(samples: np.ndarray, sample_rate: int | None = None) -> float:
    return len(samples) / float(sample_rate or settings.sample_rate)
