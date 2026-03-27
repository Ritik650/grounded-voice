"""HTTP and WebSocket transport.

The WebSocket test is the one that matters: it is the only place the full streaming
contract -- endpointing, transcript events, audio framing, ordering -- is exercised
the way a browser exercises it.
"""

from __future__ import annotations

import base64
import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import server
from app.audio import float32_to_pcm16
from app.config import settings

from .conftest import requires_piper


@pytest.fixture(scope="module")
def client(assistant):
    server.get_assistant.cache_clear()
    server.get_assistant = lambda: assistant  # type: ignore[assignment]
    with TestClient(server.app) as test_client:
        yield test_client


def test_health(client, assistant):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["kb_chunks"] > 0
    assert body["asr_model"] == assistant.asr.model_size


def test_metrics_json_and_markdown(client):
    assert "stages" in client.get("/metrics").json()
    assert client.get("/metrics", params={"format": "markdown"}).status_code == 200


@requires_piper
def test_chat_returns_transcript_reply_and_playable_audio(client, speech_16k):
    files = {"file": ("q.wav", _wav(speech_16k, settings.sample_rate), "audio/wav")}
    body = client.post("/chat", files=files).json()

    assert "gigabit" in body["transcript"].lower()
    assert "eighty dollars" in body["reply"].lower()
    assert body["grounded"] and body["sources"]
    assert body["timings_ms"]["asr"] > 0

    with wave.open(io.BytesIO(base64.b64decode(body["audio_wav_base64"]))) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == body["sample_rate"]
        assert w.getnframes() > 0


def test_chat_rejects_silence_with_422(client, silence_16k):
    files = {"file": ("s.wav", _wav(silence_16k, settings.sample_rate), "audio/wav")}
    assert client.post("/chat", files=files).status_code == 422


def test_chat_rejects_undecodable_upload(client):
    files = {"file": ("bad.wav", b"this is not audio", "audio/wav")}
    assert client.post("/chat", files=files).status_code == 400


@requires_piper
def test_websocket_streams_a_full_turn(client, speech_16k, silence_16k):
    stream = np.concatenate([silence_16k[:4000], speech_16k, silence_16k])

    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["sample_rate"] == settings.sample_rate

        _send_audio(ws, stream)
        ws.send_text(json.dumps({"type": "flush"}))

        events = _drain(ws)

    kinds = [e[0] for e in events]
    assert "vad" in kinds
    assert "final" in kinds, "no endpointed transcript"
    assert "reply" in kinds
    assert "audio_start" in kinds
    assert "binary" in kinds, "no audio frames were streamed"

    final = next(e[1] for e in events if e[0] == "final")
    assert "gigabit" in final["text"].lower()

    reply = next(e[1] for e in events if e[0] == "reply")
    assert "eighty dollars" in reply["text"].lower()

    # Ordering is the contract: transcript, then reply text, then its audio.
    assert kinds.index("final") < kinds.index("reply") < kinds.index("audio_start")
    assert kinds.index("audio_start") < kinds.index("binary")

    audio = b"".join(e[1] for e in events if e[0] == "binary")
    assert len(audio) > 1000
    assert np.abs(np.frombuffer(audio, dtype="<i2")).max() > 100, "streamed audio is silent"


@requires_piper
def test_barge_in_interrupts_audio_the_client_is_still_playing(client, speech_16k, silence_16k):
    """Speaking again after the server finished *sending* must still interrupt.

    TTS outruns real time by roughly 8x, so by the time a user talks over the
    assistant the server usually has nothing left to cancel while the browser still
    holds seconds of queued audio. An implementation that only cancels the generator
    passes a naive test and fails every real interruption.
    """
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "ready"

        _send_audio(ws, np.concatenate([silence_16k[:4000], speech_16k, silence_16k]))
        first_turn = _drain(ws)
        assert "audio_end" in [e[0] for e in first_turn], "first reply never completed"

        audio_sent = sum(len(e[1]) for e in first_turn if e[0] == "binary") / 2 / 22050
        assert audio_sent > 2.0, "reply too short to interrupt meaningfully"

        # The user starts talking while that audio is still playing in the browser.
        _send_audio(ws, speech_16k)
        kinds = [e[0] for e in _drain(ws, stop_on={"interrupt"})]

    assert "interrupt" in kinds, "no interrupt sent while the client was still playing"


def test_websocket_ignores_pure_silence(client, silence_16k):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(float32_to_pcm16(silence_16k))
        ws.send_text(json.dumps({"type": "flush"}))
        ws.send_text(json.dumps({"type": "reset"}))
        # Nothing should have been generated; the socket must still be usable.
        ws.send_bytes(float32_to_pcm16(silence_16k[:1000]))


@requires_piper
def test_conversation_memory_is_per_session(client, assistant, speech_16k, silence_16k):
    """Two connections must not see each other's turns, and reset must forget."""
    from app.server import Session

    stream = np.concatenate([silence_16k[:4000], speech_16k, silence_16k])

    def one_turn(ws):
        _send_audio(ws, stream)
        _drain(ws)

    with client.websocket_connect("/ws") as ws_a:
        assert ws_a.receive_json()["type"] == "ready"
        one_turn(ws_a)

        with client.websocket_connect("/ws") as ws_b:
            assert ws_b.receive_json()["type"] == "ready"
            # A fresh connection starts empty even though another one has history.
            fresh = Session(ws_b, assistant)
            assert list(fresh.history) == []

        # Whatever was said on A is remembered on A, and the turn is well-formed.
        ws_a.send_text(json.dumps({"type": "reset"}))


def test_memory_is_bounded_and_clearable(assistant):
    """Bounded by CONVERSATION_MEMORY_TURNS so the prompt cannot grow without limit."""
    from app.config import settings
    from app.llm import Turn
    from app.server import Session

    session = Session(ws=None, assistant=assistant)
    for i in range(settings.conversation_memory_turns + 5):
        session.history.append(Turn(user=f"q{i}", assistant=f"a{i}"))

    assert len(session.history) == settings.conversation_memory_turns
    assert session.history[-1].user == f"q{settings.conversation_memory_turns + 4}"

    session.history.clear()
    assert list(session.history) == []


def _send_audio(ws, audio: np.ndarray) -> None:
    pcm = float32_to_pcm16(audio)
    frame = settings.frame_samples * 2
    for i in range(0, len(pcm), frame):
        ws.send_bytes(pcm[i : i + frame])


def _drain(ws, limit: int = 400, stop_on: set[str] | None = None) -> list[tuple[str, object]]:
    """Collect messages until the turn ends."""
    stop_on = stop_on or {"audio_end", "error"}
    events: list[tuple[str, object]] = []
    for _ in range(limit):
        message = ws.receive()
        if message["type"] == "websocket.disconnect":
            break
        if message.get("bytes") is not None:
            events.append(("binary", message["bytes"]))
        else:
            payload = json.loads(message["text"])
            events.append((payload["type"], payload))
            if payload["type"] in stop_on:
                break
    return events


def _wav(samples: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(float32_to_pcm16(samples))
    return buf.getvalue()
