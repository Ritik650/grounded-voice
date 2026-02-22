"""Per-stage latency instrumentation.

The latency table is the headline claim of this project, so timing is built into the
pipeline rather than measured by a separate benchmark script that drifts away from
what the server actually does. Every turn -- batch or streaming -- produces a Trace,
and the same Trace type feeds both the /metrics endpoint and eval/latency_eval.py.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock

# Stage order used for reporting, so tables always read in pipeline order.
STAGE_ORDER = ["vad", "asr_queue", "asr", "retrieval", "llm", "tts_first_chunk", "tts_total"]

# The number that matters: user stops speaking -> first byte of audio plays.
END_TO_END = "end_to_end_first_audio"


@dataclass
class Trace:
    label: str = "turn"
    t0: float = field(default_factory=time.perf_counter)
    stages: dict[str, float] = field(default_factory=dict)
    marks: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            # += so a stage entered repeatedly (streaming ASR over several chunks)
            # accumulates its true cost instead of reporting only the last call.
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - start) * 1000

    def mark(self, name: str) -> float:
        """Record ms elapsed since the trace started. Idempotent -- first write wins,
        so `mark("end_to_end_first_audio")` inside a chunk loop times the *first* chunk.
        """
        elapsed = (time.perf_counter() - self.t0) * 1000
        self.marks.setdefault(name, elapsed)
        return self.marks[name]

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000

    def as_dict(self) -> dict[str, float]:
        ordered = {k: round(self.stages[k], 1) for k in STAGE_ORDER if k in self.stages}
        ordered.update({k: round(v, 1) for k, v in self.stages.items() if k not in ordered})
        ordered.update({k: round(v, 1) for k, v in self.marks.items()})
        return ordered


class LatencyRegistry:
    """Rolling window of recent traces, for percentile reporting."""

    def __init__(self, maxlen: int = 500):
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=maxlen))
        self._lock = Lock()
        self.turns = 0

    def record(self, trace: Trace) -> None:
        with self._lock:
            self.turns += 1
            for name, ms in {**trace.stages, **trace.marks}.items():
                self._samples[name].append(ms)

    def summary(self) -> dict[str, dict[str, float]]:
        with self._lock:
            snapshot = {k: list(v) for k, v in self._samples.items() if v}

        def order(name: str) -> tuple[int, str]:
            return (STAGE_ORDER.index(name) if name in STAGE_ORDER else len(STAGE_ORDER), name)

        return {
            name: {
                "n": len(vals),
                "median_ms": round(statistics.median(vals), 1),
                "p95_ms": round(_percentile(vals, 95), 1),
                "mean_ms": round(statistics.fmean(vals), 1),
            }
            for name, vals in sorted(snapshot.items(), key=lambda kv: order(kv[0]))
        }

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self.turns = 0


def _percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    # Nearest-rank; with the small n an interactive demo produces, interpolation
    # would imply precision the sample size doesn't support.
    rank = max(1, min(len(ordered), int(round(pct / 100.0 * len(ordered)))))
    return ordered[rank - 1]


def format_table(summary: dict[str, dict[str, float]]) -> str:
    """Markdown table, ready to paste into the README."""
    if not summary:
        return "_No latency samples recorded._"
    rows = ["| Stage | n | Median (ms) | p95 (ms) |", "|---|---|---|---|"]
    for name, s in summary.items():
        label = f"**{name}**" if name == END_TO_END else name
        rows.append(f"| {label} | {s['n']} | {s['median_ms']} | {s['p95_ms']} |")
    return "\n".join(rows)


registry = LatencyRegistry()
