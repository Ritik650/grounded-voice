"""Voice activity detection and utterance segmentation.

Silero VAD run directly on ONNX Runtime -- the `silero-vad` PyPI package pulls in
torch + torchaudio (~2.5 GB) purely to wrap a 2 MB ONNX file, which is a bad trade
for a CPU service that has no other torch dependency.

Two jobs, and they are genuinely different:

  1. *Endpointing* -- decide when the user has finished talking, so ASR gets a whole
     phrase instead of a fragment. This is a hysteresis problem: a naive
     probability > threshold test chops utterances at every inter-word pause.
  2. *Barge-in* -- notice the user talking over the assistant, fast enough to stop
     playback before it feels rude.

Both fall out of one frame-level state machine, so both live here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import settings


class EventType(StrEnum):
    SPEECH_START = "speech_start"  # confirmed onset -- the barge-in trigger
    UTTERANCE = "utterance"  # confirmed offset -- a complete phrase for ASR


@dataclass
class VadEvent:
    type: EventType
    audio: np.ndarray | None = None  # populated for UTTERANCE
    start_s: float = 0.0
    end_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class SileroVAD:
    """Thin frame-level wrapper: 512 samples in, speech probability out.

    Two undocumented requirements of the raw ONNX graph, both of which fail silently:

    * Each call must be fed the last 64 samples of the *previous* window prepended to
      the current one (576 samples total at 16 kHz). The graph declares its input as
      dynamic, so feeding a bare 512-sample window raises nothing -- it just returns
      near-zero probability for everything. Measured on clear speech: mean p = 0.002
      without the context, 0.970 with it.
    * The model is a stateful LSTM, so `reset()` between utterances matters; carrying
      state across a turn degrades the first frames of the next one.
    """

    # 64 samples at 16 kHz, 32 at 8 kHz -- matches Silero's own wrapper.
    CONTEXT_SAMPLES = {16_000: 64, 8_000: 32}

    def __init__(self, model_path: Path | None = None, sample_rate: int | None = None):
        self.sample_rate = sample_rate or settings.sample_rate
        path = Path(model_path or settings.vad_model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Silero VAD model missing at {path}. Run: python scripts/download_models.py"
            )

        opts = ort.SessionOptions()
        # One thread is faster here: the model is tiny and we call it every 32 ms, so
        # thread pool wake-up dominates any parallelism win.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}
        self.window = 512 if self.sample_rate == 16_000 else 256
        self.context_size = self.CONTEXT_SAMPLES.get(self.sample_rate, 64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        """Speech probability for exactly one window of samples."""
        if len(frame) != self.window:
            # Pad a short tail rather than dropping it -- the final frame of an
            # utterance is often partial and often the one carrying the offset.
            frame = np.pad(frame, (0, self.window - len(frame)))

        window = frame.reshape(1, -1).astype(np.float32)
        inputs = {
            "input": np.concatenate([self._context, window], axis=1),
            "sr": np.array(self.sample_rate, dtype=np.int64),
            "state": self._state,
        }
        inputs = {k: v for k, v in inputs.items() if k in self._input_names}
        out, self._state = self.session.run(None, inputs)
        self._context = window[:, -self.context_size :]
        return float(out[0][0])


class UtteranceSegmenter:
    """Turns a stream of arbitrary-sized PCM chunks into complete utterances.

    Callers push whatever the socket delivered; the segmenter re-frames to Silero's
    fixed window internally and returns any events that fired.
    """

    def __init__(self, vad: SileroVAD | None = None, **overrides):
        self.vad = vad or SileroVAD()
        # A recycled VAD (session reset) still holds the previous utterance's LSTM
        # state, which skews the first few frames of the new one.
        self.vad.reset()
        self.sample_rate = self.vad.sample_rate
        self.window = self.vad.window
        frame_ms = 1000.0 * self.window / self.sample_rate

        cfg = {
            "threshold": settings.vad_threshold,
            "min_speech_ms": settings.vad_min_speech_ms,
            "min_silence_ms": settings.vad_min_silence_ms,
            "speech_pad_ms": settings.vad_speech_pad_ms,
            "max_utterance_ms": settings.vad_max_utterance_ms,
            **overrides,
        }
        self.threshold = cfg["threshold"]
        # Exiting speech uses a lower bar than entering it. Without this asymmetry
        # the tail of a word (trailing fricatives, decaying vowels) dips under the
        # threshold and clips the last syllable off the transcript.
        self.exit_threshold = max(0.15, self.threshold - 0.15)
        self._min_speech_frames = max(1, int(cfg["min_speech_ms"] / frame_ms))
        self._min_silence_frames = max(1, int(cfg["min_silence_ms"] / frame_ms))
        self._max_frames = max(1, int(cfg["max_utterance_ms"] / frame_ms))
        self._preroll = deque(maxlen=max(1, int(cfg["speech_pad_ms"] / frame_ms)))

        self._leftover = np.zeros(0, dtype=np.float32)
        self._frames: list[np.ndarray] = []
        self._speech_run = 0
        self._silence_run = 0
        self._in_speech = False
        self._frames_seen = 0
        self._start_frame = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def in_hangover(self) -> bool:
        """Speaking, but currently inside a pause that may turn out to be the endpoint."""
        return self._in_speech and self._silence_run > 0

    @property
    def current_audio(self) -> np.ndarray:
        """Audio buffered for the utterance in progress, including pre-roll.

        Lets the partial-transcript decoder work from exactly the audio the final
        decode will see, rather than re-accumulating it and losing the pre-roll.
        """
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._frames)

    def push(self, samples: np.ndarray) -> list[VadEvent]:
        events: list[VadEvent] = []
        buf = np.concatenate([self._leftover, samples.astype(np.float32, copy=False)])

        n_full = len(buf) // self.window
        for i in range(n_full):
            frame = buf[i * self.window : (i + 1) * self.window]
            events.extend(self._consume_frame(frame))
        self._leftover = buf[n_full * self.window :]
        return events

    def flush(self) -> list[VadEvent]:
        """End of stream: emit whatever is buffered if it is long enough to be real."""
        events: list[VadEvent] = []
        if self._leftover.size:
            events.extend(self._consume_frame(self._leftover))
            self._leftover = np.zeros(0, dtype=np.float32)
        if self._in_speech and len(self._frames) >= self._min_speech_frames:
            events.append(self._close_utterance())
        else:
            self._discard()
        return events

    # -- internals ------------------------------------------------------------

    def _consume_frame(self, frame: np.ndarray) -> list[VadEvent]:
        events: list[VadEvent] = []
        prob = self.vad.probability(frame)
        self._frames_seen += 1

        bar = self.exit_threshold if self._in_speech or self._speech_run else self.threshold
        is_speech = prob >= bar

        if is_speech:
            self._speech_run += 1
            self._silence_run = 0
            if not self._frames:
                # Candidate onset: seed with pre-roll so the first phoneme survives.
                self._frames = list(self._preroll)
                self._start_frame = self._frames_seen - len(self._frames)
            self._frames.append(frame)

            if not self._in_speech and self._speech_run >= self._min_speech_frames:
                self._in_speech = True
                events.append(
                    VadEvent(EventType.SPEECH_START, start_s=self._to_seconds(self._start_frame))
                )
        else:
            self._silence_run += 1
            self._speech_run = 0
            if self._frames:
                self._frames.append(frame)  # keep the hangover; it holds word tails
                if self._in_speech and self._silence_run >= self._min_silence_frames:
                    events.append(self._close_utterance())
                elif not self._in_speech and self._silence_run >= self._min_silence_frames:
                    self._discard()  # blip that never became speech
            else:
                self._preroll.append(frame)

        if self._in_speech and len(self._frames) >= self._max_frames:
            events.append(self._close_utterance())
        return events

    def _close_utterance(self) -> VadEvent:
        audio = np.concatenate(self._frames) if self._frames else np.zeros(0, dtype=np.float32)
        event = VadEvent(
            EventType.UTTERANCE,
            audio=audio,
            start_s=self._to_seconds(self._start_frame),
            end_s=self._to_seconds(self._frames_seen),
        )
        self._discard()
        return event

    def _discard(self) -> None:
        self._frames = []
        self._preroll.clear()
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self.vad.reset()

    def _to_seconds(self, frame_index: int) -> float:
        return frame_index * self.window / self.sample_rate


def segment_audio(samples: np.ndarray, **overrides) -> list[VadEvent]:
    """Offline convenience: run the same segmenter over a whole array."""
    seg = UtteranceSegmenter(**overrides)
    events = seg.push(samples)
    events.extend(seg.flush())
    return [e for e in events if e.type is EventType.UTTERANCE]
