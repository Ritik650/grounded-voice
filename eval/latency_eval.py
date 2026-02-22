"""Per-stage latency profiling.

Measures the real pipeline, not a mock: questions are synthesized to audio with Piper,
then pushed through the same code path the server uses, so ASR is decoding an actual
waveform rather than being handed text.

Two numbers are reported and they mean different things:

  batch     end_to_end = the whole reply is transcribed, answered and fully synthesized
            before anything is returned. This is /chat.
  streaming end_to_end_first_audio = user stops speaking -> first audio frame is on the
            wire. This is /ws, and it is the number that describes how the assistant
            *feels*. It is lower because the first sentence is spoken while the rest is
            still being generated.

Usage:
    python -m eval.latency_eval --runs 5
    python -m eval.latency_eval --runs 5 --compare tiny.en base.en small.en
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.asr import ASR  # noqa: E402
from app.config import ROOT, settings  # noqa: E402
from app.metrics import END_TO_END, Trace, format_table, registry  # noqa: E402
from app.pipeline import VoiceAssistant  # noqa: E402
from app.vad import SileroVAD  # noqa: E402

QUESTIONS = [
    "How much does the gigabit plan cost?",
    "What happens if I pay my bill late?",
    "My router light is blinking amber, what should I do?",
    "Is there a fee for cancelling my contract early?",
    "When does support close in the evening?",
    "How long does a standard installation take?",
]


def synthesize_questions(assistant: VoiceAssistant, questions: list[str]) -> list[np.ndarray]:
    """Render each question to 16 kHz audio, the format the mic would deliver."""
    from app.audio import resample

    audios = []
    for q in questions:
        pcm = assistant.tts.synthesize(q).astype(np.float32) / 32767.0
        audios.append(resample(pcm, assistant.tts.sample_rate, settings.sample_rate))
    return audios


def profile_vad(vad: SileroVAD, audio: np.ndarray, runs: int = 200) -> float:
    """Median milliseconds to classify one 32 ms frame."""
    frame = audio[: vad.window]
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        vad.probability(frame)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def profile_batch(assistant: VoiceAssistant, audios: list[np.ndarray], runs: int) -> None:
    for _ in range(runs):
        for audio in audios:
            assistant.respond(audio)


def profile_streaming(assistant: VoiceAssistant, questions: list[str], runs: int) -> list[float]:
    """Drive the streaming path and time the first audio frame out."""
    first_audio_ms = []
    for _ in range(runs):
        for question in questions:
            trace = Trace(label="stream")
            for _sentence, frame in assistant.stream_reply(question, trace):
                if frame is not None:
                    break  # first audio frame: stop the clock, drop the rest
            if END_TO_END in trace.marks:
                first_audio_ms.append(trace.marks[END_TO_END])
    return first_audio_ms


def run(model_size: str | None, runs: int, skip_streaming: bool = False) -> dict:
    registry.reset()
    assistant = VoiceAssistant(asr=ASR(model_size=model_size) if model_size else None)
    assistant.bootstrap()

    audios = synthesize_questions(assistant, QUESTIONS)
    vad_ms = profile_vad(SileroVAD(), audios[0])

    profile_batch(assistant, audios, runs)
    batch_summary = registry.summary()

    streaming = {}
    if not skip_streaming:
        registry.reset()
        first_audio = profile_streaming(assistant, QUESTIONS, runs)
        streaming = registry.summary()
        if first_audio:
            streaming["end_to_end_first_audio"] = {
                "n": len(first_audio),
                "median_ms": round(statistics.median(first_audio), 1),
                "p95_ms": round(sorted(first_audio)[int(0.95 * (len(first_audio) - 1))], 1),
                "mean_ms": round(statistics.fmean(first_audio), 1),
            }

    return {
        "asr_model": assistant.asr.model_size,
        "compute_type": assistant.asr.compute_type,
        "llm_backend": assistant.llm_backend,
        "tts_engine": assistant.tts.engine.name,
        "turns": runs * len(QUESTIONS),
        "vad_frame_ms": round(vad_ms, 3),
        "batch": batch_summary,
        "streaming": streaming,
    }


def print_report(result: dict) -> None:
    print(f"\n{'=' * 68}")
    print(
        f"ASR {result['asr_model']} ({result['compute_type']})  |  "
        f"LLM {result['llm_backend']}  |  TTS {result['tts_engine']}  |  "
        f"{result['turns']} turns"
    )
    print("=" * 68)
    print(
        f"\nVAD, per 32 ms frame: {result['vad_frame_ms']:.2f} ms "
        f"({32 / result['vad_frame_ms']:.0f}x faster than real time)\n"
    )
    print("Batch turn (/chat) -- full synthesis before response\n")
    print(format_table(result["batch"]))
    if result["streaming"]:
        print("\nStreaming turn (/ws) -- time to first audio frame\n")
        print(format_table(result["streaming"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="passes over the question set")
    parser.add_argument("--compare", nargs="*", help="ASR model sizes to profile in turn")
    parser.add_argument("--skip-streaming", action="store_true")
    parser.add_argument("--json", type=Path, default=ROOT / "eval" / "results" / "latency.json")
    args = parser.parse_args()

    results = []
    for model in args.compare or [None]:
        result = run(model, args.runs, args.skip_streaming)
        print_report(result)
        results.append(result)

    if args.compare and len(results) > 1:
        print(f"\n{'=' * 68}\nModel comparison -- median ms\n{'=' * 68}")
        print("| Model | ASR | Batch end-to-end | Streaming first audio |")
        print("|---|---|---|---|")
        for r in results:
            asr_ms = r["batch"].get("asr", {}).get("median_ms", "-")
            e2e = r["batch"].get(END_TO_END, {}).get("median_ms", "-")
            stream = r["streaming"].get("end_to_end_first_audio", {}).get("median_ms", "-")
            print(f"| {r['asr_model']} | {asr_ms} | {e2e} | {stream} |")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
