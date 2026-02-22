from __future__ import annotations

import numpy as np

from app.metrics import END_TO_END, Trace, format_table, registry

from .conftest import requires_piper


@requires_piper
def test_full_turn_produces_a_grounded_spoken_answer(assistant, speech_16k):
    result = assistant.respond(speech_16k)

    assert "gigabit" in result.transcript.lower()
    assert "eighty dollars" in result.reply.lower()
    assert result.sources
    assert result.grounded
    assert result.audio is not None and result.audio.size > 0
    assert np.abs(result.audio).max() > 0.01, "synthesized reply is silent"


@requires_piper
def test_turn_records_every_stage(assistant, speech_16k):
    result = assistant.respond(speech_16k)
    for stage in ("asr", "retrieval", "llm", "tts_total", END_TO_END):
        assert stage in result.timings, f"missing timing for {stage}"
    assert result.timings[END_TO_END] > 0


def test_silence_yields_an_empty_turn_without_calling_the_llm(assistant, silence_16k):
    result = assistant.respond(silence_16k)
    assert result.transcript == ""
    assert result.reply == ""
    assert "llm" not in result.timings


@requires_piper
def test_streaming_yields_sentences_before_audio(assistant):
    pairs = list(assistant.stream_reply("how much is the gigabit plan"))

    assert pairs, "no output from the streaming path"
    assert pairs[0][1] is None, "audio arrived before its sentence was announced"
    assert any(frame is not None for _, frame in pairs)

    frames = [f for _, f in pairs if f is not None]
    assert all(f.sample_rate == assistant.tts.sample_rate for f in frames)
    assert all(len(f.pcm) % 2 == 0 for f in frames), "a frame split a 16-bit sample"


@requires_piper
def test_streaming_speaks_sentence_one_while_the_llm_is_still_writing(assistant, monkeypatch):
    """The actual pipelining win: TTS overlaps generation.

    Measured separately, Piper's own `synthesize()` already yields audio sentence by
    sentence (first frame ~76 ms whether the text is passed whole or split), so
    splitting text before handing it over buys nothing on its own. What the
    sentence-level pipeline buys is overlap with a *slow* generator -- a hosted LLM
    streaming tokens over the network. This models that with a backend that pauses
    between sentences, which is the only way to assert the property deterministically
    rather than racing two similar TTS calls and reading the noise.
    """
    import time as _time

    from app import llm

    gap_s = 0.5

    class SlowStreamingBackend:
        name = "slow"

        def stream(self, question, context):
            for sentence in (
                "Support runs from seven to eleven. ",
                "Outage cover is always on. ",
                "You can also use the portal.",
            ):
                yield sentence
                _time.sleep(gap_s)

        def complete(self, question, context):
            return "".join(self.stream(question, context))

    monkeypatch.setitem(llm.BACKENDS, "slow", SlowStreamingBackend)
    llm.get_llm.cache_clear()
    monkeypatch.setattr(assistant, "llm_backend", "slow")

    trace = Trace()
    start = _time.perf_counter()
    for _sentence, frame in assistant.stream_reply("support hours", trace):
        if frame is not None:
            break
    first_audio_s = _time.perf_counter() - start
    llm.get_llm.cache_clear()

    # Generation alone takes 3 * gap_s. Audio must be playing long before that.
    assert first_audio_s < 2 * gap_s, (
        f"first audio took {first_audio_s:.2f}s; generation alone is {3 * gap_s:.2f}s"
    )
    assert END_TO_END in trace.marks


def test_trace_accumulates_repeated_stages():
    trace = Trace()
    for _ in range(3):
        with trace.stage("tts_total"):
            pass
    assert trace.stages["tts_total"] >= 0

    trace.mark("first")
    first = trace.marks["first"]
    trace.mark("first")
    assert trace.marks["first"] == first, "mark() should keep the earliest value"


def test_registry_summarises_and_formats():
    registry.reset()
    for _ in range(4):
        trace = Trace()
        with trace.stage("asr"):
            pass
        trace.mark(END_TO_END)
        registry.record(trace)

    summary = registry.summary()
    assert summary["asr"]["n"] == 4
    assert "Median" in format_table(summary)
    registry.reset()
