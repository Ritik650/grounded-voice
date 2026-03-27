from __future__ import annotations

import numpy as np

from app.asr import ASR, StreamingTranscriber, _is_hallucination
from app.config import settings

from .conftest import requires_piper


@requires_piper
def test_transcribes_synthesized_speech(asr, speech_16k):
    result = asr.transcribe(speech_16k)
    words = result.text.lower()
    assert "gigabit" in words
    assert "cost" in words or "month" in words
    assert result.duration_s > 1.0


def test_empty_audio_returns_empty_transcript(asr):
    assert asr.transcribe(np.zeros(500, dtype=np.float32)).text == ""


def test_silence_does_not_hallucinate(asr, silence_16k):
    """Whisper invents subtitle boilerplate on silence; the guard must catch it."""
    assert asr.transcribe(silence_16k).text == ""


def test_hallucination_filter():
    assert _is_hallucination("Thank you.", no_speech_prob=0.9)
    assert _is_hallucination("yeah yeah yeah yeah yeah yeah yeah yeah yeah", 0.1)
    assert not _is_hallucination("Thank you.", no_speech_prob=0.1)
    assert not _is_hallucination("The gigabit plan costs eighty dollars.", 0.1)


@requires_piper
def test_streaming_transcriber_throttles_decodes(asr, speech_16k):
    stream = StreamingTranscriber(asr, min_new_audio_s=1.0)
    half_second = settings.sample_rate // 2

    stream.add_audio(speech_16k[:half_second])
    assert stream.partial() is None, "decoded before a full second of new audio arrived"

    stream.add_audio(speech_16k[half_second : half_second * 3])
    assert stream.partial() is not None


@requires_piper
def test_final_resets_the_buffer(asr, speech_16k):
    stream = StreamingTranscriber(asr)
    stream.add_audio(speech_16k)
    assert stream.final().text
    assert stream.buffer.size == 0


def test_model_metadata_is_exposed():
    asr = ASR()
    assert asr.model_size and asr.compute_type and asr.device
