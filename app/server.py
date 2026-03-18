"""FastAPI transport layer: batch /chat (M1) and streaming /ws (M2).

## WebSocket protocol

Client -> server
  binary                      raw PCM16LE mono @ `sample_rate` (see the `ready` message)
  {"type": "flush"}           end of input; force-close any in-flight utterance
  {"type": "reset"}           drop buffered audio and cancel any reply in flight

Server -> client
  {"type": "ready", ...}      handshake: sample rates and model names
  {"type": "vad", "state": "speech_start" | "speech_end"}
  {"type": "partial", "text"} provisional transcript; may be rewritten
  {"type": "final", "text"}   endpointed transcript; this is what drives the answer
  {"type": "reply", "text", "sources"}
  {"type": "audio_start", "sample_rate"}   binary frames follow, at *this* rate
  binary                      PCM16LE mono TTS audio
  {"type": "audio_end"}
  {"type": "interrupt"}       barge-in: stop playback and discard buffered audio
  {"type": "timings", ...}    per-stage latency for the turn just completed

## Concurrency

Receiving and responding are separate tasks. That is the whole point: if the socket
loop blocked while a reply was being spoken, the server would be deaf for exactly the
window in which the user is most likely to interrupt. Model calls are blocking C
extensions, so they run in the executor; the response thread checks a cancel flag
between audio frames, which bounds barge-in reaction time to one frame (~200 ms).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .asr import StreamingTranscriber
from .audio import load_audio, pcm16_to_float32, wav_bytes
from .config import ROOT, settings
from .llm import Turn
from .metrics import Trace, format_table, registry
from .pipeline import VoiceAssistant, get_assistant
from .vad import EventType, UtteranceSegmenter

log = logging.getLogger(__name__)
FRONTEND_DIR = ROOT / "frontend"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    assistant = get_assistant()
    # Ingestion and warmup are seconds of blocking CPU; keep the event loop free so
    # health checks answer while the container is still coming up.
    await asyncio.to_thread(assistant.bootstrap)
    log.info("Assistant ready (kb chunks=%d)", assistant.kb.count())
    yield


app = FastAPI(title="Real-Time Voice Assistant", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    assistant = get_assistant()
    return {
        "status": "ok",
        "asr_model": assistant.asr.model_size,
        "tts_engine": assistant.tts.engine.name,
        "llm_backend": assistant.llm_backend,
        "kb_chunks": assistant.kb.count(),
    }


@app.get("/metrics")
async def metrics(format: str = "json"):
    summary = registry.summary()
    if format == "markdown":
        return Response(format_table(summary), media_type="text/markdown")
    return {"turns": registry.turns, "stages": summary}


@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    """M1: upload spoken audio, get transcript + grounded reply + spoken answer."""
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    tmp = ROOT / "data" / "tmp" / f"upload{suffix}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(await file.read())

    try:
        audio = await asyncio.to_thread(load_audio, tmp)
    except Exception as exc:
        raise HTTPException(400, f"Could not decode audio: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)

    result = await asyncio.to_thread(get_assistant().respond, audio)
    if not result.transcript:
        return JSONResponse(
            {"transcript": "", "reply": "", "error": "no speech detected"}, status_code=422
        )

    payload = {
        "transcript": result.transcript,
        "reply": result.reply,
        "sources": result.sources,
        "grounded": result.grounded,
        "timings_ms": result.timings,
        "sample_rate": result.sample_rate,
    }
    if result.audio is not None:
        wav = wav_bytes(result.audio, result.sample_rate)
        payload["audio_wav_base64"] = base64.b64encode(wav).decode("ascii")
    return payload


@app.post("/chat/wav")
async def chat_wav(file: UploadFile = File(...)):
    """Same turn, but returns the spoken reply as a plain WAV (easy to curl)."""
    response = await chat(file)
    if isinstance(response, JSONResponse):
        return response
    audio_b64 = response.get("audio_wav_base64")
    if not audio_b64:
        raise HTTPException(500, "No audio was synthesized")
    return Response(
        base64.b64decode(audio_b64),
        media_type="audio/wav",
        headers={
            "X-Transcript": response["transcript"],
            "X-Reply": response["reply"],
        },
    )


# --------------------------------------------------------------------------------
# Streaming session
# --------------------------------------------------------------------------------


class Session:
    """One WebSocket conversation."""

    def __init__(self, ws: WebSocket, assistant: VoiceAssistant):
        self.ws = ws
        self.assistant = assistant
        self.segmenter = UtteranceSegmenter()
        self.partials = StreamingTranscriber(assistant.asr, initial_prompt=assistant.asr_prompt)

        # Conversation memory lives on the session, never at module scope: two callers
        # sharing one process must not see each other's turns. A bounded deque also
        # means a long conversation cannot grow the prompt without limit.
        self.history: deque[Turn] = deque(maxlen=max(0, settings.conversation_memory_turns))

        self._send_lock = asyncio.Lock()  # one writer at a time on the socket
        self._asr_lock = asyncio.Lock()  # the Whisper model is not re-entrant
        self._partial_running = False
        # Rolling cost of a partial decode on *this* machine. Partials are a nicety;
        # the final transcript is the product. On hardware where a partial cannot
        # finish inside the end-of-utterance hangover, starting one guarantees the
        # answer queues behind it -- measured at 856 ms of added dead air in a
        # CPU-limited container. So the machine's own speed decides whether partials
        # are affordable, rather than a constant tuned on a fast laptop.
        self._partial_ms = 0.0
        self._response: asyncio.Task | None = None
        self._cancel = threading.Event()  # read by the response thread between frames

        # Playback bookkeeping. TTS runs ~8x faster than real time, so the server
        # finishes *sending* a reply long before the user finishes *hearing* it.
        # Cancelling the generator is therefore not enough to implement barge-in:
        # most interruptions arrive when there is nothing left to cancel and several
        # seconds of audio are still queued in the browser. Tracking how much audio
        # was sent, and when, tells us whether the client is probably still playing.
        self._playback_started: float | None = None
        self._audio_sent_s = 0.0

    # -- messaging ------------------------------------------------------------

    async def send_json(self, payload: dict) -> None:
        async with self._send_lock:
            await self.ws.send_json(payload)

    async def send_bytes(self, data: bytes) -> None:
        async with self._send_lock:
            await self.ws.send_bytes(data)

    # -- input ----------------------------------------------------------------

    async def on_audio(self, raw: bytes) -> None:
        samples = pcm16_to_float32(raw)
        events = self.segmenter.push(samples)

        if self.segmenter.in_speech:
            self.partials.set_audio(self.segmenter.current_audio)

        for event in events:
            if event.type is EventType.SPEECH_START:
                await self.send_json({"type": "vad", "state": "speech_start"})
                await self.barge_in()
            elif event.type is EventType.UTTERANCE:
                await self.send_json({"type": "vad", "state": "speech_end"})
                await self.on_utterance(event.audio)

        # Never start a partial once the user has stopped making sound. Partials and
        # the final transcript share one Whisper model, so a partial launched into the
        # end-of-utterance pause is still decoding when the endpoint fires, and the
        # answer waits behind it. Measured cost of getting this wrong: ~250 ms of dead
        # air on every turn. Starting them only during active speech means any partial
        # in flight has the whole 600 ms hangover to finish in.
        if (
            self.segmenter.in_speech
            and not self.segmenter.in_hangover
            and not self._partial_running
            and self._partial_ms <= self._partial_budget_ms
        ):
            # Set the guard here, not inside the task: audio frames arrive faster than
            # the loop schedules tasks, so a flag set in the coroutine would let dozens
            # of decodes queue up behind one another.
            self._partial_running = True
            asyncio.create_task(self._emit_partial())

    @property
    def _partial_budget_ms(self) -> float:
        """How long a partial may take and still be certain to finish before the
        endpoint fires. 80% of the hangover leaves margin for scheduling jitter."""
        return 0.8 * settings.vad_min_silence_ms

    async def _emit_partial(self) -> None:
        try:
            if not self.partials.should_decode():
                return
            async with self._asr_lock:
                started = time.monotonic()
                text = await asyncio.to_thread(self.partials.partial)
                elapsed_ms = (time.monotonic() - started) * 1000

            # Track the worst recent cost rather than an average: one slow decode is
            # enough to land on the endpoint, and decaying toward the fast case would
            # keep re-learning that the hard way.
            self._partial_ms = max(elapsed_ms, 0.7 * self._partial_ms)
            if self._partial_ms > self._partial_budget_ms:
                log.info(
                    "Disabling partial transcripts: %.0f ms per decode exceeds the "
                    "%.0f ms endpoint budget on this machine.",
                    self._partial_ms,
                    self._partial_budget_ms,
                )
            if text:
                await self.send_json({"type": "partial", "text": text})
        except Exception:
            log.exception("partial transcript failed")
        finally:
            self._partial_running = False

    async def on_utterance(self, audio) -> None:
        trace = Trace(label="stream_turn")
        # Timed separately so contention shows up in /metrics instead of hiding inside
        # an inflated ASR figure.
        with trace.stage("asr_queue"):
            await self._asr_lock.acquire()
        try:
            transcript = await asyncio.to_thread(lambda: self.assistant.transcribe(audio, trace))
        finally:
            self._asr_lock.release()
        self.partials.reset()

        if not transcript:
            return
        await self.send_json({"type": "final", "text": transcript})
        self._response = asyncio.create_task(self.respond(transcript, trace))

    # -- output ---------------------------------------------------------------

    async def respond(self, question: str, trace: Trace) -> None:
        self._cancel.clear()
        loop = asyncio.get_running_loop()
        spoken: list[str] = []

        def produce() -> None:
            """Runs in the executor: pulls the pipeline generator, pushes to the socket.

            Blocking on each send gives backpressure -- generation cannot outrun the
            socket, which keeps memory bounded on a slow connection.
            """
            started = False
            # The pipeline yields (sentence, None) once when a sentence is ready, then
            # (sentence, frame) for each of its audio frames.
            for sentence, frame in self.assistant.stream_reply(
                question, trace, history=list(self.history)
            ):
                if self._cancel.is_set():
                    break
                if frame is None:
                    spoken.append(sentence)
                    _block_on(loop, self._announce(sentence, started))
                    started = True
                else:
                    self._audio_sent_s += frame.duration_s
                    _block_on(loop, self.send_bytes(frame.pcm))
            if started and not self._cancel.is_set():
                _block_on(loop, self.send_json({"type": "audio_end"}))

        try:
            await loop.run_in_executor(None, produce)
            await self.send_json({"type": "timings", **trace.as_dict()})
        except asyncio.CancelledError:
            self._cancel.set()
            raise
        except Exception:
            log.exception("response failed")
            with contextlib.suppress(Exception):
                await self.send_json({"type": "error", "message": "response failed"})
        finally:
            # Recorded even when barge-in cut the reply short: the user heard the part
            # that was spoken, so a follow-up pronoun may well refer to it. Storing
            # nothing would leave the next turn resolving against a gap.
            if spoken and self.history.maxlen:
                self.history.append(Turn(user=question, assistant=" ".join(spoken)))

    async def _announce(self, sentence: str, already_started: bool) -> None:
        await self.send_json({"type": "reply", "text": sentence})
        if not already_started:
            self._playback_started = time.monotonic()
            self._audio_sent_s = 0.0
            await self.send_json(
                {"type": "audio_start", "sample_rate": self.assistant.tts.sample_rate}
            )

    @property
    def likely_playing(self) -> bool:
        """Is the client probably still playing audio we already sent?

        An estimate, deliberately: the alternative is a client-side `playback_done`
        message, which adds a round trip to the interrupt path and still lies whenever
        a packet is delayed. Erring toward "yes" is the cheap direction -- a spurious
        interrupt on an idle client is a no-op there.
        """
        if self._playback_started is None:
            return False
        return (time.monotonic() - self._playback_started) < self._audio_sent_s + 0.3

    async def barge_in(self) -> None:
        """User started talking over the assistant: stop speaking, now."""
        task = self._response
        generating = task is not None and not task.done()
        if not generating and not self.likely_playing:
            return

        if generating:
            self._cancel.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            self._response = None

        # Sent even when there is nothing left to generate: the audio the user is
        # talking over is already buffered in the browser, and only the client can
        # stop it.
        self._playback_started = None
        await self.send_json({"type": "interrupt"})

    async def reset(self) -> None:
        """Protocol-level clear: drop buffered audio, cancel the reply, forget the
        conversation. Closing the socket does the same by discarding the Session."""
        await self.barge_in()
        self.segmenter = UtteranceSegmenter(vad=self.segmenter.vad)
        self.partials.reset()
        self.history.clear()

    async def flush(self) -> None:
        for event in self.segmenter.flush():
            if event.type is EventType.UTTERANCE:
                await self.send_json({"type": "vad", "state": "speech_end"})
                await self.on_utterance(event.audio)

    async def close(self) -> None:
        self._cancel.set()
        if self._response and not self._response.done():
            self._response.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._response


def _block_on(loop: asyncio.AbstractEventLoop, coro) -> None:
    """Run a coroutine on the event loop from a worker thread and wait for it."""
    asyncio.run_coroutine_threadsafe(coro, loop).result()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    assistant = get_assistant()
    session = Session(websocket, assistant)

    await session.send_json(
        {
            "type": "ready",
            "sample_rate": settings.sample_rate,
            "tts_sample_rate": assistant.tts.sample_rate,
            "asr_model": assistant.asr.model_size,
            "llm_backend": assistant.llm_backend,
        }
    )

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                await session.on_audio(data)
            elif (text := message.get("text")) is not None:
                import json

                command = json.loads(text).get("type")
                if command == "flush":
                    await session.flush()
                elif command == "reset":
                    await session.reset()
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("websocket session failed")
    finally:
        await session.close()


# Serve the demo UI last so it can't shadow the API routes above.
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def index():
        return FileResponse(FRONTEND_DIR / "index.html")
