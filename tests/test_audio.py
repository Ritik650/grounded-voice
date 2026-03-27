from __future__ import annotations

import io
import wave

import numpy as np

from app.audio import (
    duration_s,
    float32_to_pcm16,
    load_audio,
    pcm16_to_float32,
    resample,
    wav_bytes,
    write_wav,
)


def test_pcm_roundtrip_is_lossless_within_quantisation():
    original = np.linspace(-0.9, 0.9, 1000, dtype=np.float32)
    restored = pcm16_to_float32(float32_to_pcm16(original))
    assert np.allclose(original, restored, atol=1 / 32767)


def test_odd_byte_count_is_tolerated():
    """A socket frame can split mid-sample; that must not raise."""
    assert len(pcm16_to_float32(b"\x00\x01\x02")) == 1


def test_conversion_clips_instead_of_wrapping():
    """Without clipping, an over-range float wraps to full-scale noise."""
    loud = np.array([2.0, -2.0], dtype=np.float32)
    assert np.all(np.abs(pcm16_to_float32(float32_to_pcm16(loud))) <= 1.0)


def test_resample_changes_length_proportionally():
    signal = np.sin(np.linspace(0, 100, 22050)).astype(np.float32)
    out = resample(signal, 22050, 16000)
    assert abs(len(out) - 16000) < 50


def test_resample_is_a_noop_at_the_same_rate():
    signal = np.zeros(10, dtype=np.float32)
    assert resample(signal, 16000, 16000) is not None
    assert len(resample(signal, 16000, 16000)) == 10


def test_wav_bytes_are_well_formed():
    with wave.open(io.BytesIO(wav_bytes(np.zeros(1600, dtype=np.float32), 16000))) as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 1600


def test_write_then_load_preserves_the_signal(tmp_path):
    signal = (np.sin(np.linspace(0, 50, 16000)) * 0.5).astype(np.float32)
    path = write_wav(tmp_path / "out.wav", signal, 16000)
    assert np.allclose(load_audio(path, 16000), signal, atol=1e-3)


def test_stereo_is_downmixed_to_mono(tmp_path):
    import soundfile as sf

    stereo = np.zeros((1600, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, 16000)

    mono = load_audio(path, 16000)
    assert mono.ndim == 1
    assert np.allclose(mono, 0.25, atol=1e-3)


def test_duration_helper():
    assert duration_s(np.zeros(8000, dtype=np.float32), 16000) == 0.5
