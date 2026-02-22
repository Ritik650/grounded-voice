"""Word Error Rate measurement.

Two datasets, and the distinction between them matters:

  librispeech  Real human read speech (LibriSpeech test-clean). This is the only
               number worth quoting, because it is comparable to published results.
  synthetic    Piper speaks the KB, Whisper transcribes it back. Runs offline in
               seconds, so CI can gate on it -- but TTS output is unrealistically
               clean, so its WER is far below what real users produce. It detects
               regressions; it does not describe accuracy.

Usage:
    python -m eval.wer_eval --dataset librispeech --limit 100
    python -m eval.wer_eval --dataset librispeech --limit 100 --model small.en
    python -m eval.wer_eval --dataset synthetic --max-wer 0.15
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.asr import ASR  # noqa: E402
from app.audio import load_audio  # noqa: E402
from app.config import ROOT  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"

# LibriSpeech references are uppercase and unpunctuated; Whisper emits cased, punctuated
# text. Comparing them raw would report ~80% WER measuring nothing but formatting.
NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}


@dataclass
class Sample:
    audio_path: Path
    reference: str
    utterance_id: str = ""


@dataclass
class Result:
    wer: float
    cer: float
    n_samples: int
    audio_seconds: float
    compute_seconds: float
    rtf: float
    median_ms: float
    worst: list[tuple[float, str, str]]

    def report(self) -> str:
        lines = [
            f"Samples          {self.n_samples}",
            f"Audio            {self.audio_seconds / 60:.1f} min",
            f"WER              {self.wer:.2%}",
            f"CER              {self.cer:.2%}",
            f"Real-time factor {self.rtf:.3f}x  (lower is faster than real time)",
            f"Median decode    {self.median_ms:.0f} ms/utterance",
        ]
        if self.worst:
            lines.append("\nWorst utterances:")
            for wer, ref, hyp in self.worst:
                lines.append(f"  WER {wer:.0%}\n    ref: {ref}\n    hyp: {hyp}")
        return "\n".join(lines)


# -- datasets ---------------------------------------------------------------------


def load_librispeech(limit: int | None = None, subset: str = "test-clean") -> list[Sample]:
    root = EVAL_DIR / "LibriSpeech" / subset
    if not root.exists():
        _ensure_librispeech(subset)

    samples: list[Sample] = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt_id, _, text = line.partition(" ")
            flac = trans.parent / f"{utt_id}.flac"
            if flac.exists():
                samples.append(Sample(flac, text.strip(), utt_id))
        if limit and len(samples) >= limit:
            break
    return samples[:limit] if limit else samples


def _ensure_librispeech(subset: str) -> None:
    archive = EVAL_DIR / f"{subset}.tar.gz"
    if not archive.exists():
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading LibriSpeech {subset} (~346 MB)...")
        urllib.request.urlretrieve(LIBRISPEECH_URL, archive)  # noqa: S310
    print("Extracting...")
    with tarfile.open(archive) as tar:
        tar.extractall(EVAL_DIR, filter="data")


def load_synthetic(limit: int | None = None) -> list[Sample]:
    """Speak KB sentences with Piper, then transcribe them back."""
    from app.audio import write_wav
    from app.config import settings
    from app.tts import TTS

    out_dir = EVAL_DIR / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    tts = TTS()

    sentences: list[str] = []
    for doc in sorted(Path(settings.kb_dir).glob("*.md")):
        for raw in re.split(r"(?<=[.!?])\s+", doc.read_text(encoding="utf-8")):
            s = " ".join(raw.split())
            if s.startswith(("#", "*")) or not (40 <= len(s) <= 160):
                continue
            sentences.append(s)
    sentences = sentences[:limit] if limit else sentences

    samples = []
    for i, sentence in enumerate(sentences):
        path = out_dir / f"kb_{i:03d}.wav"
        if not path.exists():
            audio = tts.synthesize(sentence).astype("float32") / 32767.0
            write_wav(path, audio, tts.sample_rate)
        samples.append(Sample(path, sentence, f"kb_{i:03d}"))
    return samples


def load_manifest(path: Path, limit: int | None = None) -> list[Sample]:
    """JSONL: {"audio": "path/to.wav", "text": "reference transcript"}"""
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        audio = Path(row["audio"])
        samples.append(Sample(audio if audio.is_absolute() else path.parent / audio, row["text"]))
    return samples[:limit] if limit else samples


# -- evaluation -------------------------------------------------------------------


def normalize_reference(text: str) -> str:
    """Spell out digits so "30" vs "thirty" isn't scored as an error.

    Whisper writes numerals, the KB (and human references) often spell them out.
    That is a formatting difference, not a recognition error.
    """
    return re.sub(r"\b\d\b|\b10\b", lambda m: NUMBER_WORDS[m.group()], text)


def evaluate(samples: list[Sample], asr: ASR, show_worst: int = 3) -> Result:
    refs, hyps, durations, times = [], [], [], []

    for i, sample in enumerate(samples, 1):
        audio = load_audio(sample.audio_path)
        start = time.perf_counter()
        hypothesis = asr.transcribe(audio).text
        times.append((time.perf_counter() - start) * 1000)
        durations.append(len(audio) / 16_000)
        refs.append(normalize_reference(sample.reference))
        hyps.append(normalize_reference(hypothesis))
        if i % 25 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)}", end="\r", flush=True)
    print()

    measures = jiwer.process_words(
        refs, hyps, reference_transform=NORMALIZE, hypothesis_transform=NORMALIZE
    )
    cer = jiwer.cer([r.lower() for r in refs], [h.lower() for h in hyps])

    per_sample = sorted(
        (
            (jiwer.wer(r, h, reference_transform=NORMALIZE, hypothesis_transform=NORMALIZE), r, h)
            for r, h in zip(refs, hyps, strict=True)
            if r.strip()
        ),
        key=lambda t: -t[0],
    )

    audio_seconds = sum(durations)
    compute_seconds = sum(times) / 1000
    return Result(
        wer=measures.wer,
        cer=float(cer),
        n_samples=len(samples),
        audio_seconds=audio_seconds,
        compute_seconds=compute_seconds,
        rtf=compute_seconds / audio_seconds if audio_seconds else 0.0,
        median_ms=statistics.median(times) if times else 0.0,
        worst=per_sample[:show_worst],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="librispeech", choices=["librispeech", "synthetic", "manifest"]
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", help="override ASR model size, e.g. tiny.en / small.en")
    parser.add_argument("--max-wer", type=float, help="exit non-zero above this WER (CI gate)")
    parser.add_argument("--json", type=Path, help="write results here")
    args = parser.parse_args()

    if args.dataset == "librispeech":
        samples = load_librispeech(args.limit)
    elif args.dataset == "synthetic":
        samples = load_synthetic(args.limit)
    else:
        samples = load_manifest(args.manifest, args.limit)

    if not samples:
        print("No samples found.")
        return 1

    asr = ASR(model_size=args.model) if args.model else ASR()
    print(f"Dataset {args.dataset} | model {asr.model_size} | {asr.compute_type} on {asr.device}\n")
    result = evaluate(samples, asr)
    print(result.report())

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "model": asr.model_size,
                    "compute_type": asr.compute_type,
                    "n_samples": result.n_samples,
                    "wer": round(result.wer, 4),
                    "cer": round(result.cer, 4),
                    "rtf": round(result.rtf, 4),
                    "median_decode_ms": round(result.median_ms, 1),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if args.max_wer is not None and result.wer > args.max_wer:
        print(f"\nFAIL: WER {result.wer:.2%} exceeds threshold {args.max_wer:.2%}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
