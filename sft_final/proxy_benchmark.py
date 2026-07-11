#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
proxy_benchmark.py — measure batching quality WITHOUT real training.

Idea (confirmed design):
  * The GPU processes every token in the padded tensor (real + pad). So the work,
    and therefore the time, of one training step is proportional to the number of
    *padded* tokens B_i * T_padded_i. Real tokens are constant across batching
    strategies, so any time difference between two batching methods comes entirely
    from padding.
  * PROXY TRAINING TIME for one epoch = sum over batches of:
        cost_i = alpha * (B_i * T_padded_i)  [+ gamma * B_i * T_padded_i^2]  + beta
    With alpha=1, beta=gamma=0 this is just the total padded-token count — a
    relative measure, perfect for comparing "ours" vs the baseline. Set alpha/beta
    (optionally fit from a few real steps) to get an estimate in seconds.

This module never loads a model. It drives the C++ pipeline for one epoch,
reads each pool's token counts (computed in the collator), and aggregates:
  padding %, proxy training time, useful-token ratio, per-category stats,
  scheduler health (empty-queue alerts, fallbacks, skipped categories),
  and prefetch formation/stall.

Run it twice — once with the round-robin category scheduler ("ours") and once
with baseline=True (length-agnostic random batching, the "previous method") —
and compare().
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


BAND_NAMES = ["SHORT", "MEDIUM", "LONG", "CHUNKED"]


@dataclass
class ProxyModel:
    """Cost model cost = alpha * padded_tokens [+ gamma * padded_tokens^2] + beta."""
    alpha: float = 1.0       # per padded-token cost (1.0 => relative units)
    beta: float = 0.0        # fixed per-step overhead
    gamma: float = 0.0       # optional quadratic (attention) term, per (padded_tokens^2)

    def cost(self, padded_tokens: int) -> float:
        c = self.alpha * padded_tokens + self.beta
        if self.gamma:
            c += self.gamma * float(padded_tokens) ** 2
        return c


@dataclass
class EpochStats:
    method: str
    pools: int = 0
    samples: int = 0
    padded_tokens: int = 0
    real_tokens: int = 0
    pad_tokens: int = 0
    proxy_time: float = 0.0
    wall_seconds: float = 0.0
    # scheduler health
    empty_alerts: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    fallback_pools: int = 0
    skipped_categories: int = 0
    samples_streamed: int = 0
    # prefetch
    formation_total_s: float = 0.0
    stall_total_s: float = 0.0
    # per-category padded/pad tokens (profile_index -> [padded, pad, pools])
    per_cat: Dict[int, List[int]] = field(default_factory=dict)

    @property
    def padding_pct(self) -> float:
        return 100.0 * self.pad_tokens / self.padded_tokens if self.padded_tokens else 0.0

    @property
    def useful_pct(self) -> float:
        return 100.0 - self.padding_pct

    def summary(self) -> Dict:
        return {
            "method": self.method,
            "pools": self.pools,
            "samples": self.samples,
            "padded_tokens": self.padded_tokens,
            "real_tokens": self.real_tokens,
            "pad_tokens": self.pad_tokens,
            "padding_pct": round(self.padding_pct, 2),
            "useful_pct": round(self.useful_pct, 2),
            "proxy_time": round(self.proxy_time, 2),
            "wall_seconds": round(self.wall_seconds, 3),
            "fallback_pools": self.fallback_pools,
            "skipped_categories": self.skipped_categories,
            "empty_alerts": {BAND_NAMES[i]: self.empty_alerts[i] for i in range(4)},
            "samples_streamed": self.samples_streamed,
            "formation_total_s": round(self.formation_total_s, 3),
            "stall_total_s": round(self.stall_total_s, 3),
            "per_category": {
                f"CAT{ci}": {
                    "pools": v[2],
                    "padded_tokens": v[0],
                    "pad_tokens": v[1],
                    "padding_pct": round(100.0 * v[1] / v[0], 2) if v[0] else 0.0,
                }
                for ci, v in sorted(self.per_cat.items())
            },
        }

def run_epoch_proxy(
    pipeline,
    model: Optional[ProxyModel] = None,
    method: str = "round_robin",
    max_pools: Optional[int] = None,
    idle_grace: int = 200,
) -> EpochStats:
    """Drive the C++ pipeline for ONE epoch (no model) and aggregate proxy stats.

    `pipeline` is a started uds_loader.DataPipeline. `method` label is inferred.
    The loop ends when the streamer is done AND the queues yield no more pools
    (empty pools observed for `idle_grace` consecutive polls while streamer_done).
    """
    model = model or ProxyModel()
    st = EpochStats(method=method)
    t0 = time.perf_counter()

    idle = 0
    while True:
        pool = pipeline.next_pool()
        if pool is None or len(pool) == 0:
            # nothing right now; stop only once streamer is done and it stays empty
            if pipeline.streamer_done():
                idle += 1
                if idle >= idle_grace:
                    break
            else:
                idle = 0
            time.sleep(0.001)
            continue
        idle = 0

        pt = int(pool.padded_tokens)
        st.pools += 1
        st.samples += int(pool.batch_size)
        st.padded_tokens += pt
        st.real_tokens += int(pool.real_tokens)
        st.pad_tokens += int(pool.pad_tokens)
        st.proxy_time += model.cost(pt)

        ci = int(pool.profile_index)
        cat = st.per_cat.setdefault(ci, [0, 0, 0])
        cat[0] += pt
        cat[1] += int(pool.pad_tokens)
        cat[2] += 1

        if max_pools and st.pools >= max_pools:
            break

    st.wall_seconds = time.perf_counter() - t0
    st.empty_alerts = [int(pipeline.empty_alerts(b)) for b in range(4)]
    st.fallback_pools = int(pipeline.fallback_pools())
    try:
        st.skipped_categories = int(pipeline.skipped_categories())
    except Exception:
        st.skipped_categories = 0
    st.samples_streamed = int(pipeline.samples_streamed())
    st.formation_total_s = float(pipeline.formation_total_s())
    st.stall_total_s = float(pipeline.stall_total_s())
    return st


def compare(ours: EpochStats, baseline: EpochStats) -> str:
    """Human-readable ours-vs-baseline comparison table."""
    def red(a, b):  # % reduction of a vs b
        return (100.0 * (b - a) / b) if b else 0.0

    lines = []
    lines.append("=" * 68)
    lines.append("  PROXY BENCHMARK — length-aware categories vs baseline")
    lines.append("=" * 68)
    lines.append(f"  {'metric':<26}{'ours':>14}{'baseline':>14}{'delta':>12}")
    lines.append("-" * 68)
    lines.append(f"  {'padding %':<26}{ours.padding_pct:>14.2f}{baseline.padding_pct:>14.2f}"
                 f"{red(ours.pad_tokens, baseline.pad_tokens):>11.1f}%")
    lines.append(f"  {'padded tokens':<26}{ours.padded_tokens:>14,}{baseline.padded_tokens:>14,}"
                 f"{red(ours.padded_tokens, baseline.padded_tokens):>11.1f}%")
    lines.append(f"  {'proxy train time':<26}{ours.proxy_time:>14,.0f}{baseline.proxy_time:>14,.0f}"
                 f"{red(ours.proxy_time, baseline.proxy_time):>11.1f}%")
    lines.append(f"  {'real (useful) tokens':<26}{ours.real_tokens:>14,}{baseline.real_tokens:>14,}"
                 f"{'—':>12}")
    lines.append(f"  {'samples':<26}{ours.samples:>14,}{baseline.samples:>14,}{'—':>12}")
    lines.append(f"  {'pools (batches)':<26}{ours.pools:>14,}{baseline.pools:>14,}{'—':>12}")
    lines.append("-" * 68)
    lines.append(f"  {'fallback pools (ours)':<26}{ours.fallback_pools:>14,}")
    lines.append(f"  {'skipped categories':<26}{ours.skipped_categories:>14,}")
    lines.append(f"  {'empty alerts S/M/L/C':<26}"
                 f"{'/'.join(str(x) for x in ours.empty_alerts):>14}")
    lines.append("=" * 68)
    lines.append(f"  RESULT: length-aware batching cut padded tokens by "
                 f"{red(ours.padded_tokens, baseline.padded_tokens):.1f}% and "
                 f"proxy train time by {red(ours.proxy_time, baseline.proxy_time):.1f}%.")
    lines.append("=" * 68)
    return "\n".join(lines)
