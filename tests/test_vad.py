"""VAD and endpointing.

The behaviours worth locking down are the ones that are wrong in a naive
implementation: not firing on room tone, not splitting one sentence into three
utterances at its internal pauses, and not depending on the size of the chunks the
socket happens to deliver.
"""

from __future__ import annotations

import numpy as np

from app.config import settings
from app.vad import EventType, SileroVAD, UtteranceSegmenter, segment_audio

from .conftest import requires_piper, requires_vad

pytestmark = requires_vad


def test_silence_produces_no_utterance(silence_16k):
    assert segment_audio(silence_16k) == []


@requires_piper
def test_speech_produces_exactly_one_utterance(speech_16k, silence_16k):
    stream = np.concatenate([silence_16k[:8000], speech_16k, silence_16k])
    utterances = segment_audio(stream)

    assert len(utterances) == 1
    # Endpointed audio should cover the speech, plus pre-roll and hangover padding,
    # without having swallowed the full second of trailing silence.
    spoken = len(speech_16k) / settings.sample_rate
    assert spoken <= utterances[0].duration_s <= spoken + 1.2


@requires_piper
def test_internal_pauses_do_not_split_an_utterance(tts):
    """A comma-length pause must not endpoint -- this is the hysteresis check."""
    from app.audio import resample

    pcm = tts.synthesize("First, the router restarts. Then the light turns white.")
    speech = resample(pcm.astype(np.float32) / 32767.0, tts.sample_rate, settings.sample_rate)
    assert len(segment_audio(speech)) == 1


@requires_piper
def test_chunk_size_does_not_change_the_result(speech_16k, silence_16k):
    """Identical audio must segment identically whether it arrives in 100-sample or
    5000-sample chunks. Frame reassembly is where streaming VADs usually break."""
    stream = np.concatenate([silence_16k[:4000], speech_16k, silence_16k])

    results = []
    for chunk in (100, 512, 1237, 5000):
        segmenter = UtteranceSegmenter()
        events = []
        for i in range(0, len(stream), chunk):
            events.extend(segmenter.push(stream[i : i + chunk]))
        events.extend(segmenter.flush())
        utterances = [e for e in events if e.type is EventType.UTTERANCE]
        results.append(len(utterances))

    assert len(set(results)) == 1, f"segmentation varied with chunk size: {results}"


@requires_piper
def test_speech_start_precedes_utterance(speech_16k, silence_16k):
    """Barge-in depends on SPEECH_START arriving while the user is still talking."""
    segmenter = UtteranceSegmenter()
    stream = np.concatenate([silence_16k[:4000], speech_16k, silence_16k])

    events = segmenter.push(stream) + segmenter.flush()
    types = [e.type for e in events]

    assert EventType.SPEECH_START in types
    assert types.index(EventType.SPEECH_START) < types.index(EventType.UTTERANCE)


def test_probability_accepts_a_short_final_frame():
    vad = SileroVAD()
    assert 0.0 <= vad.probability(np.zeros(200, dtype=np.float32)) <= 1.0


@requires_piper
def test_max_utterance_length_forces_a_cut(speech_16k):
    """A talker who never pauses must not buffer without bound."""
    continuous = np.tile(speech_16k, 6)  # ~15 s with no real silence
    segmenter = UtteranceSegmenter(max_utterance_ms=2000)

    utterances = [e for e in segmenter.push(continuous) if e.type is EventType.UTTERANCE]

    assert utterances, "continuous speech produced no utterance at all"
    assert all(u.duration_s <= 3.0 for u in utterances), (
        f"cap not enforced: {[round(u.duration_s, 2) for u in utterances]}"
    )
