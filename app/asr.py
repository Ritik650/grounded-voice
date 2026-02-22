"""Speech recognition: faster-whisper (CTranslate2) wrapped for batch and streaming use.

Whisper is not a streaming model -- it consumes a fixed 30 s window and emits text for
all of it. Two consequences drive the design here:

  * Utterance-level ASR (the final transcript) is triggered by the VAD's endpoint, not
    by a timer. That gives Whisper a complete phrase, which is what it is good at.
  * Partial transcripts are re-decodes of the growing buffer. They are *provisional*
    and can rewrite earlier words as more context arrives -- fine for on-screen
    feedback, never used to trigger the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from .audio import load_audio
from .config import settings

# Whisper is trained on 30 s windows that always contain speech, so on near-silence it
# confabulates -- overwhelmingly with the phrases below (subtitle-corpus artefacts).
HALLUCINATION_PHRASES = {
    "thank you.",
    "thanks for watching!",
    "thank you for watching.",
    "you",
    ".",
    "bye.",
    "please subscribe.",
    "subtitles by the amara.org community",
}


@dataclass
class Transcript:
    text: str
    language: str = settings.asr_language
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    duration_s: float = 0.0

    def __bool__(self) -> bool:
        return bool(self.text.strip())


class ASR:
    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ):
        self.model_size = model_size or settings.asr_model_size
        self.device = device or settings.asr_device
        self.compute_type = compute_type or settings.asr_compute_type
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            num_workers=1,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        beam_size: int | None = None,
        initial_prompt: str | None = None,
    ) -> Transcript:
        """Transcribe float32 mono @ 16 kHz."""
        if audio.size < settings.sample_rate // 10:  # < 100 ms is never a word
            return Transcript(text="")

        segments, info = self.model.transcribe(
            audio.astype(np.float32, copy=False),
            language=settings.asr_language,
            beam_size=beam_size if beam_size is not None else settings.asr_beam_size,
            # Our own segmenter already removed the silence; letting Whisper's VAD cut
            # again would trim word onsets a second time.
            vad_filter=False,
            # Each utterance is decoded independently. Carrying context across turns
            # sounds appealing but is the main driver of runaway repetition loops.
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=initial_prompt,
        )

        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments).strip()
        avg_logprob = float(np.mean([s.avg_logprob for s in segments])) if segments else 0.0
        no_speech = float(np.mean([s.no_speech_prob for s in segments])) if segments else 1.0

        if _is_hallucination(text, no_speech):
            text = ""

        return Transcript(
            text=text,
            language=info.language if info else settings.asr_language,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech,
            duration_s=len(audio) / settings.sample_rate,
        )

    def transcribe_file(self, path: str | Path, **kwargs) -> Transcript:
        return self.transcribe(load_audio(path), **kwargs)


def _is_hallucination(text: str, no_speech_prob: float) -> bool:
    stripped = text.strip().lower()
    if not stripped:
        return True
    if stripped in HALLUCINATION_PHRASES and no_speech_prob > 0.5:
        return True
    # A single token repeated to the end of the window is the classic loop failure.
    words = stripped.split()
    return len(words) > 8 and len(set(words)) <= 2


class StreamingTranscriber:
    """Emits provisional transcripts while the user is still speaking.

    Re-decoding the whole buffer each time is O(n^2) over an utterance, which is fine
    at conversational lengths (a few seconds) and much simpler than local-agreement
    streaming. `min_new_audio_s` throttles it so a fast talker can't queue up decodes
    faster than the CPU retires them.
    """

    def __init__(self, asr: ASR, min_new_audio_s: float = 1.0, initial_prompt: str | None = None):
        self.asr = asr
        self.min_new_audio_s = min_new_audio_s
        self.initial_prompt = initial_prompt
        self.reset()

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_decoded_len = 0
        self._last_text = ""

    @property
    def buffer(self) -> np.ndarray:
        return self._buffer

    def add_audio(self, samples: np.ndarray) -> None:
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32, copy=False)])

    def set_audio(self, samples: np.ndarray) -> None:
        """Replace the buffer wholesale.

        Used by the streaming server so partials decode exactly the VAD's current
        utterance buffer -- pre-roll included -- instead of a parallel accumulation
        that starts late and clips the first word.
        """
        self._buffer = samples.astype(np.float32, copy=False)

    def should_decode(self) -> bool:
        new_samples = len(self._buffer) - self._last_decoded_len
        return new_samples >= self.min_new_audio_s * settings.sample_rate

    def partial(self) -> str | None:
        """Decode the buffer if enough new audio has arrived; None otherwise."""
        if not self.should_decode():
            return None
        self._last_decoded_len = len(self._buffer)
        # Greedy on partials: they are throwaway, and beam search would steal CPU
        # from the final decode that actually drives the answer.
        result = self.asr.transcribe(self._buffer, beam_size=1, initial_prompt=self.initial_prompt)
        if result.text and result.text != self._last_text:
            self._last_text = result.text
            return result.text
        return None

    def final(self, audio: np.ndarray | None = None) -> Transcript:
        result = self.asr.transcribe(
            self._buffer if audio is None else audio,
            beam_size=settings.asr_beam_size,
            initial_prompt=self.initial_prompt,
        )
        self.reset()
        return result
