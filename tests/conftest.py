"""Shared fixtures.

Model-backed fixtures are session-scoped -- constructing Whisper and Piper costs
seconds, and doing it per-test would make the suite too slow to run on every change.
Tests that need model files are skipped (not failed) when the files are absent, so a
fresh clone can run the logic tests before downloading anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

KB_FIXTURE = """
# Test Knowledge Base

The Gigabit plan costs eighty dollars a month and includes unlimited data.

Support is open from 7 a.m. to 11 p.m. local time, seven days a week.

Accounts more than 14 days overdue are charged a late fee of five dollars.
"""


def _have(path) -> bool:
    return Path(path).exists()


requires_piper = pytest.mark.skipif(
    not _have(settings.piper_model_path), reason="Piper voice not downloaded"
)
requires_vad = pytest.mark.skipif(
    not _have(settings.vad_model_path), reason="Silero VAD model not downloaded"
)


@pytest.fixture(scope="session")
def tts():
    from app.tts import TTS

    return TTS()


@pytest.fixture(scope="session")
def asr():
    from app.asr import ASR

    return ASR()


@pytest.fixture(scope="session")
def speech_16k(tts):
    """A real spoken utterance at the pipeline's input sample rate."""
    from app.audio import resample

    pcm = tts.synthesize("How much does the Gigabit plan cost per month?")
    return resample(pcm.astype(np.float32) / 32767.0, tts.sample_rate, settings.sample_rate)


@pytest.fixture
def silence_16k():
    # Not pure zeros: real "silence" has a noise floor, and a VAD that only rejects
    # digital black is a VAD that fires on room tone.
    rng = np.random.default_rng(0)
    return (rng.standard_normal(settings.sample_rate) * 0.001).astype(np.float32)


@pytest.fixture(scope="session")
def kb(tmp_path_factory):
    from app.rag import KnowledgeBase

    kb_dir = tmp_path_factory.mktemp("kb")
    (kb_dir / "test.md").write_text(KB_FIXTURE, encoding="utf-8")
    knowledge = KnowledgeBase(url=":memory:", collection="test_kb")
    knowledge.ingest_directory(kb_dir)
    return knowledge


@pytest.fixture(scope="session")
def assistant(asr, kb, tts):
    from app.pipeline import VoiceAssistant

    va = VoiceAssistant(asr=asr, kb=kb, tts=tts, llm_backend="extractive")
    va.refresh_asr_prompt()
    return va
