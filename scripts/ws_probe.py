"""Diagnostic WebSocket client: drives a real server and prints the event timeline.

Kept in the repo because it is the fastest way to see what the streaming protocol
actually did, including barge-in, without a browser.

    uvicorn app.server:app --port 8000
    python scripts/ws_probe.py [--barge-in]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.audio import float32_to_pcm16, resample  # noqa: E402
from app.config import settings  # noqa: E402


def make_speech(text: str) -> np.ndarray:
    from app.tts import TTS

    tts = TTS()
    pcm = tts.synthesize(text).astype(np.float32) / 32767.0
    return resample(pcm, tts.sample_rate, settings.sample_rate)


async def send_audio(ws, audio: np.ndarray, realtime: bool = True) -> None:
    """Push audio in 32 ms frames, optionally paced like a real microphone."""
    pcm = float32_to_pcm16(audio)
    frame = settings.frame_samples * 2
    for i in range(0, len(pcm), frame):
        await ws.send(pcm[i : i + frame])
        if realtime:
            await asyncio.sleep(settings.frame_ms / 1000)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    parser.add_argument("--text", default="How much does the gigabit plan cost per month?")
    parser.add_argument(
        "--barge-in",
        action="store_true",
        help="interrupt the reply mid-sentence with a second utterance",
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    print("Synthesizing input audio...")
    question = make_speech(args.text)
    interrupt = make_speech("Actually, what are your support hours?") if args.barge_in else None
    silence = np.zeros(int(settings.sample_rate * 1.0), dtype=np.float32)

    async with websockets.connect(args.url, max_size=None) as ws:
        start = time.perf_counter()
        audio_bytes = 0
        first_audio_at = None
        speech_end_at = None

        # Without barge-in one reply ends the run; with it we expect the first reply,
        # an interrupt, then the reply to the interrupting question.
        turns_expected = 2 if args.barge_in else 1

        async def reader():
            nonlocal audio_bytes, first_audio_at, speech_end_at
            turns_done = 0
            async for message in ws:
                elapsed = (time.perf_counter() - start) * 1000
                if isinstance(message, bytes):
                    audio_bytes += len(message)
                    if first_audio_at is None:
                        first_audio_at = elapsed
                        print(f"[{elapsed:7.0f} ms] <audio starts>")
                    continue
                event = json.loads(message)
                kind = event.pop("type")
                print(f"[{elapsed:7.0f} ms] {kind:12} {event if event else ''}")
                if kind == "vad" and event.get("state") == "speech_end":
                    speech_end_at = elapsed
                # "timings" is the last message of a turn, after audio_end.
                if kind == "timings":
                    turns_done += 1
                    if turns_done >= turns_expected:
                        return

        reader_task = asyncio.create_task(reader())

        await send_audio(ws, np.concatenate([silence[:8000], question, silence]))
        if interrupt is not None:
            # Wait for the reply to start, then talk over it.
            while first_audio_at is None and not reader_task.done():
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.4)
            print("--- speaking over the assistant ---")
            await send_audio(ws, np.concatenate([interrupt, silence]))
        else:
            await ws.send(json.dumps({"type": "flush"}))

        try:
            await asyncio.wait_for(reader_task, timeout=args.timeout)
        except TimeoutError:
            print(f"\nTIMED OUT after {args.timeout}s waiting for the turn to finish")
            reader_task.cancel()
            return 1

        rate = 22050
        print(
            f"\nReceived {audio_bytes} bytes of audio ({audio_bytes / 2 / rate:.2f}s at {rate} Hz)"
        )
        if first_audio_at and speech_end_at:
            # The metric that describes how the assistant feels: the pause the user
            # actually sits through. Wall-clock from connect is meaningless here --
            # most of it is the user talking.
            print(f"RESPONSE_LATENCY_MS {first_audio_at - speech_end_at:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
