"""Turn orchestration: audio in -> transcript -> grounded answer -> audio out.

Holds the models (they are expensive to construct and safe to share) and the wiring
between stages, so the FastAPI layer stays a transport shim and the eval harness can
drive the exact same code path the server does.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import llm
from .asr import ASR
from .audio import load_audio
from .config import settings
from .metrics import END_TO_END, Trace, registry
from .rag import Chunk, KnowledgeBase
from .tts import TTS, AudioFrame

log = logging.getLogger(__name__)


@dataclass
class TurnResult:
    transcript: str
    reply: str
    sources: list[str] = field(default_factory=list)
    audio: np.ndarray | None = None
    sample_rate: int = 22_050
    timings: dict[str, float] = field(default_factory=dict)
    grounded: bool = True


class VoiceAssistant:
    def __init__(
        self,
        asr: ASR | None = None,
        kb: KnowledgeBase | None = None,
        tts: TTS | None = None,
        llm_backend: str | None = None,
    ):
        self.asr = asr or ASR()
        self.kb = kb or KnowledgeBase()
        self.tts = tts or TTS()
        self.llm_backend = llm_backend or settings.llm_backend
        self._asr_prompt: str | None = None

    # -- setup ----------------------------------------------------------------

    def bootstrap(self, ingest: bool = True) -> None:
        """Load the KB and pay every model's first-inference cost up front."""
        if ingest and self.kb.count() == 0:
            self.kb.ingest_directory()
        self.refresh_asr_prompt()
        self.warmup()

    def refresh_asr_prompt(self) -> None:
        """Bias ASR toward the KB's proper nouns.

        Grounding starts at recognition: if Whisper writes "Aurora Bank" as "a roar
        of bank", no amount of retrieval quality recovers the turn.
        """
        hint = self.kb.vocabulary_hint()
        self._asr_prompt = f"Terms that may be mentioned: {hint}." if hint else None

    @property
    def asr_prompt(self) -> str | None:
        """Domain vocabulary bias, shared with the streaming partial decoder."""
        return self._asr_prompt

    def warmup(self) -> None:
        self.asr.transcribe(np.zeros(settings.sample_rate // 2, dtype=np.float32))
        self.tts.warmup()
        self._warmup_llm()

    def _warmup_llm(self) -> None:
        """Open the TLS connection and settle the thinking-budget question up front.

        A hosted backend otherwise spends the user's first turn on a TLS handshake and,
        for models that reject `thinking_budget`, one rejected request before the retry
        succeeds. Costs a single API call at boot; on the free tier that is one request
        out of the per-minute quota, spent where nobody is waiting.
        """
        if self.llm_backend == "extractive":
            return
        try:
            start = time.perf_counter()
            llm.answer("ping", "", backend=self.llm_backend)
            log.info("LLM warmup (%s) took %.2fs", self.llm_backend, time.perf_counter() - start)
        except Exception as exc:
            # Never block startup on a remote service. Turns fall back to extractive.
            log.warning("LLM warmup failed for %r: %s", self.llm_backend, exc)

    # -- stages ---------------------------------------------------------------

    def transcribe(self, audio: np.ndarray, trace: Trace) -> str:
        with trace.stage("asr"):
            return self.asr.transcribe(audio, initial_prompt=self._asr_prompt).text

    def retrieve(self, question: str, trace: Trace) -> tuple[str, list[Chunk]]:
        with trace.stage("retrieval"):
            return self.kb.context_for(question)

    def generate(self, question: str, context: str, trace: Trace) -> str:
        with trace.stage("llm"):
            return llm.answer(question, context, backend=self.llm_backend)

    def generate_sentences(self, question: str, context: str) -> Iterator[str]:
        return llm.stream_sentences(question, context, backend=self.llm_backend)

    # -- whole turns ----------------------------------------------------------

    def respond(self, audio: np.ndarray, synthesize: bool = True) -> TurnResult:
        """M1: one complete batch turn."""
        trace = Trace(label="batch_turn")
        transcript = self.transcribe(audio, trace)

        if not transcript:
            trace.mark(END_TO_END)
            registry.record(trace)
            return TurnResult(
                transcript="",
                reply="",
                timings=trace.as_dict(),
                sample_rate=self.tts.sample_rate,
            )

        context, chunks = self.retrieve(transcript, trace)
        reply = self.generate(transcript, context, trace)

        speech = None
        if synthesize and reply:
            with trace.stage("tts_total"):
                speech = self.tts.synthesize(reply).astype(np.float32) / 32767.0
        trace.mark(END_TO_END)
        registry.record(trace)

        return TurnResult(
            transcript=transcript,
            reply=reply,
            sources=sorted({c.source for c in chunks}),
            audio=speech,
            sample_rate=self.tts.sample_rate,
            timings=trace.as_dict(),
            grounded=bool(chunks) and reply != llm.NO_CONTEXT_REPLY,
        )

    def respond_file(self, wav_in: str | Path, wav_out: str | Path | None = None) -> TurnResult:
        result = self.respond(load_audio(wav_in))
        if wav_out and result.audio is not None:
            from .audio import write_wav

            write_wav(wav_out, result.audio, result.sample_rate)
        return result

    def stream_reply(
        self, question: str, trace: Trace | None = None
    ) -> Iterator[tuple[str, AudioFrame | None]]:
        """M2: sentence-by-sentence generation, each sentence synthesized as it lands.

        Yields (sentence, frame) pairs -- sentence first with frame=None so the caller
        can display the text immediately, then its audio frames. The generator is the
        cancellation point for barge-in: the caller simply stops iterating.
        """
        trace = trace or Trace(label="stream_turn")
        with trace.stage("retrieval"):
            context, _ = self.kb.context_for(question)

        # For a streaming turn the useful LLM number is time-to-first-sentence, not
        # total generation time: everything after the first sentence overlaps with TTS
        # and playback, so it never reaches the user as latency.
        llm_start = time.perf_counter()
        try:
            for sentence in self.generate_sentences(question, context):
                trace.stages.setdefault("llm", (time.perf_counter() - llm_start) * 1000)
                yield sentence, None
                with trace.stage("tts_total"):
                    for frame in self.tts.stream(sentence):
                        trace.mark("tts_first_chunk")
                        trace.mark(END_TO_END)
                        yield sentence, frame
        finally:
            trace.stages.setdefault("llm", (time.perf_counter() - llm_start) * 1000)
            registry.record(trace)


@lru_cache(maxsize=1)
def get_assistant() -> VoiceAssistant:
    return VoiceAssistant()


def run(wav_in: str, wav_out: str) -> dict:
    """Plan-compatible one-shot entry point."""
    result = get_assistant().respond_file(wav_in, wav_out)
    return {
        "transcript": result.transcript,
        "reply": result.reply,
        "audio": str(wav_out),
        "sources": result.sources,
        "timings": result.timings,
    }
