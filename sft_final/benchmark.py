#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py — timing harness for the pipeline.

Tracks the three headline numbers requested:
    * batch formation time   (time the training loop waits for a ready pool,
                              i.e. the STALL; plus the raw C++ formation cost,
                              most of which is hidden behind training)
    * training time          (scoring + forward + backward + optimizer step)
    * total time             (wall clock)

Also derives throughput and prefetch overlap efficiency so you can see how much
of formation the prefetcher successfully hid.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Dict, Optional


class Benchmark:
    def __init__(self, sync: Optional[Callable[[], None]] = None):
        # sync() is called before stopping a GPU-bound phase (e.g. cuda.synchronize)
        self._sync = sync or (lambda: None)
        self._totals: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._samples_seen = 0
        self._tokens_seen = 0
        self._t_start = None
        self._t_end = None

    def start(self):
        self._t_start = time.perf_counter()

    def stop(self):
        self._t_end = time.perf_counter()

    @contextmanager
    def phase(self, name: str, gpu: bool = False):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if gpu:
                self._sync()
            self._totals[name] += time.perf_counter() - t0
            self._counts[name] += 1

    def add_samples(self, n: int, tokens: int = 0):
        self._samples_seen += n
        self._tokens_seen += tokens

    def total_wall(self) -> float:
        if self._t_start is None:
            return 0.0
        end = self._t_end if self._t_end is not None else time.perf_counter()
        return end - self._t_start

    def summary(self, cpp_stats: Optional[Dict[str, float]] = None) -> Dict:
        wall = self.total_wall()
        out = {
            "total_wall_s": round(wall, 4),
            "samples": self._samples_seen,
            "tokens": self._tokens_seen,
            "throughput_samples_per_s": round(self._samples_seen / wall, 2) if wall else 0,
            "throughput_tokens_per_s": round(self._tokens_seen / wall, 2) if wall else 0,
            "phases": {},
        }
        for name in sorted(self._totals):
            t = self._totals[name]
            c = self._counts[name]
            out["phases"][name] = {
                "total_s": round(t, 4),
                "mean_ms": round(1000 * t / c, 3) if c else 0,
                "pct_of_wall": round(100 * t / wall, 1) if wall else 0,
                "calls": c,
            }
        if cpp_stats:
            out["cpp_pipeline"] = cpp_stats
            # overlap efficiency: fraction of raw formation hidden by prefetch
            form = cpp_stats.get("formation_total_s", 0.0)
            stall = cpp_stats.get("stall_total_s", 0.0)
            if form > 0:
                out["prefetch_overlap_efficiency_pct"] = round(
                    100 * (1 - min(stall, form) / form), 1)
        return out

    def report(self, cpp_stats: Optional[Dict[str, float]] = None) -> str:
        s = self.summary(cpp_stats)
        lines = []
        lines.append("=" * 62)
        lines.append("  BENCHMARK SUMMARY")
        lines.append("=" * 62)
        lines.append(f"  total (wall)        : {s['total_wall_s']:>10.3f} s")
        lines.append(f"  samples trained     : {s['samples']:>10d}")
        lines.append(f"  throughput          : {s['throughput_samples_per_s']:>10.1f} samples/s"
                     f"  |  {s['throughput_tokens_per_s']:.0f} tok/s")
        lines.append("-" * 62)
        lines.append(f"  {'phase':<22}{'total_s':>10}{'mean_ms':>10}{'%wall':>8}")
        for name, p in s["phases"].items():
            lines.append(f"  {name:<22}{p['total_s']:>10.3f}{p['mean_ms']:>10.2f}{p['pct_of_wall']:>8.1f}")
        if "cpp_pipeline" in s:
            c = s["cpp_pipeline"]
            lines.append("-" * 62)
            lines.append("  C++ pipeline (batch formation):")
            lines.append(f"    raw formation total : {c.get('formation_total_s', 0):>8.3f} s"
                         f"  (mean {c.get('formation_mean_ms', 0):.2f} ms/pool)")
            lines.append(f"    consumer stall total: {c.get('stall_total_s', 0):>8.3f} s"
                         f"  (mean {c.get('stall_mean_ms', 0):.2f} ms/pool)")
            if "prefetch_overlap_efficiency_pct" in s:
                lines.append(f"    prefetch overlap    : {s['prefetch_overlap_efficiency_pct']:>7.1f} %"
                             "  (formation hidden behind training)")
        lines.append("=" * 62)
        return "\n".join(lines)

    def save(self, path: str, cpp_stats: Optional[Dict[str, float]] = None):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(cpp_stats), f, indent=2)
