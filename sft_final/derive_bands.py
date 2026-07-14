#!/usr/bin/env python3
"""
derive_bands.py — data-driven band configuration.

Reads meta.json's length_histogram and computes, from the corpus length
distribution alone:
  * the number of length bands (queues) to use,
  * their boundary cutoffs (minimizing total padding),
  * per-band population fractions (for category sampling weights),
subject to two robustness guards (starvation + resident-window feasibility).

Pipeline: histogram -> padding formula -> optimal boundaries for fixed k (DP)
          -> choose k at the elbow -> starvation guard -> window guard
          -> sampling weights -> write band_config.json.

Padding of a band spanning lengths (a, b] with upper edge b:
    pad(a,b) = C(a,b)*b - S(a,b)
where C = sample count and S = sum of real lengths in the range.

Usage:
    python derive_bands.py --meta /path/_shards/meta.json \
        --resident-window 4 --batch-size 32 --kmax 6 --elbow 0.05
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core: exact minimum-padding partition into k bands via DP.
# ---------------------------------------------------------------------------
def optimal_bands(lengths: np.ndarray, counts: np.ndarray, k: int):
    """
    lengths : sorted unique lengths, shape (n,)
    counts  : sample count at each length, shape (n,)
    k       : number of bands
    Returns (edges, total_pad) where edges = list of k band upper-edges.
    Minimizes total padding = sum over bands of  (C_band * band_max - S_band).
    Weighted by counts, so n is the number of DISTINCT lengths (<= max_seq_len).
    """
    n = len(lengths)
    k = min(k, n)

    # prefix sums over counts (C) and count*length (S) for O(1) band cost
    C = np.concatenate([[0], np.cumsum(counts)])                    # C[j] = #samples in L[0:j]
    S = np.concatenate([[0], np.cumsum(counts * lengths)])          # S[j] = real tokens in L[0:j]

    def pad(i: int, j: int) -> float:
        # band covers L[i:j], all padded up to L[j-1]
        cnt = C[j] - C[i]
        real = S[j] - S[i]
        return cnt * lengths[j - 1] - real

    INF = float("inf")
    best = np.full((n + 1, k + 1), INF)
    cut = np.zeros((n + 1, k + 1), dtype=int)
    best[0][0] = 0.0

    for b in range(1, k + 1):
        for j in range(1, n + 1):
            # last band = L[i:j]; previous b-1 bands cover L[0:i]
            lo = b - 1
            for i in range(lo, j):
                if best[i][b - 1] == INF:
                    continue
                c = best[i][b - 1] + pad(i, j)
                if c < best[j][b]:
                    best[j][b] = c
                    cut[j][b] = i

    # backtrack to recover band upper-edges
    edges: List[int] = []
    j = n
    for b in range(k, 0, -1):
        i = cut[j][b]
        edges.append(int(lengths[j - 1]))
        j = i
    edges.sort()
    return edges, float(best[n][k])


# ---------------------------------------------------------------------------
# Choose k at the elbow of the padding-vs-k curve.
# ---------------------------------------------------------------------------
def choose_k(lengths, counts, kmax: int, elbow_frac: float):
    """
    Returns (k_star, curve) where curve = {k: total_pad}.
    k_star = smallest k whose marginal improvement over k-1 falls below
    elbow_frac of the k=1 padding (i.e. adding a band stops helping).
    """
    curve = {}
    for k in range(1, kmax + 1):
        _, pad = optimal_bands(lengths, counts, k)
        curve[k] = pad
    base = curve[1] if curve[1] > 0 else 1.0
    k_star = 1
    for k in range(2, kmax + 1):
        improvement = (curve[k - 1] - curve[k]) / base
        if improvement < elbow_frac:
            k_star = k - 1
            break
        k_star = k
    return k_star, curve


# ---------------------------------------------------------------------------
# Guard 1: merge starved (thin) bands into their neighbor.
# ---------------------------------------------------------------------------
def starvation_guard(edges, lengths, counts, min_frac: float):
    """
    Drop any band holding < min_frac of all samples by removing its edge
    (merging it into the adjacent band). Recompute until all bands pass.
    """
    total = counts.sum()
    changed = True
    edges = sorted(edges)
    while changed and len(edges) > 1:
        changed = False
        # population per band defined by current edges
        lo = 0
        pops = []
        for e in edges:
            hi = np.searchsorted(lengths, e, side="right")
            pops.append(counts[lo:hi].sum())
            lo = hi
        for idx, p in enumerate(pops):
            if p / total < min_frac:
                # merge: drop this band's edge (except keep the top edge)
                drop = idx if idx < len(edges) - 1 else idx - 1
                del edges[drop]
                changed = True
                break
    return edges


# ---------------------------------------------------------------------------
# Guard 2: resident-window feasibility — cap k so each band is fed enough.
# ---------------------------------------------------------------------------
def window_feasible_k(k, resident_window, avg_shard_samples, batch_size, safety=3.0):
    """
    Each band receives ~ resident_window * avg_shard_samples / k samples per
    window. Require that to exceed safety * batch_size. Return the largest
    feasible k (<= requested k).
    """
    supply = resident_window * avg_shard_samples
    max_k = max(1, int(supply / (safety * batch_size)))
    return min(k, max_k)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def derive(meta: Dict, resident_window: int, batch_size: int,
           kmax: int, elbow_frac: float, min_band_frac: float,
           window_safety: float):
    hist: Dict[str, int] = meta["length_histogram"]
    max_seq = int(meta["max_seq_length"])
    num_shards = int(meta.get("num_shards", 1)) or 1
    num_samples = int(meta.get("num_samples", sum(int(v) for v in hist.values())))
    avg_shard_samples = num_samples / num_shards

    # histogram -> sorted arrays
    items = sorted((int(L), int(c)) for L, c in hist.items())
    lengths = np.array([L for L, _ in items], dtype=np.int64)
    counts = np.array([c for _, c in items], dtype=np.int64)

    # 1) choose k at the elbow
    k_elbow, curve = choose_k(lengths, counts, kmax, elbow_frac)

    # 2) window-feasibility cap
    k_feasible = window_feasible_k(k_elbow, resident_window, avg_shard_samples,
                                   batch_size, window_safety)

    # 3) optimal boundaries for the chosen k
    edges, total_pad = optimal_bands(lengths, counts, k_feasible)

    # 4) starvation guard (may reduce band count further)
    edges = starvation_guard(edges, lengths, counts, min_band_frac)

    # 5) per-band populations -> sampling weights; +1 CHUNKED band for > max_seq
    total = counts.sum()
    lo = 0
    band_defs = []
    for bi, e in enumerate(sorted(edges)):
        hi = np.searchsorted(lengths, e, side="right")
        pop = int(counts[lo:hi].sum())
        band_defs.append({"band": bi, "max_len": int(e),
                          "count": pop, "weight": pop / total})
        lo = hi

    n_len_bands = len(band_defs)
    chunked_band = n_len_bands  # chunked = its own single band index

    # emit config the pipeline consumes: one single-band category per band,
    # plus a chunked category.
    profile_bands = [[b["band"], b["band"]] for b in band_defs] + [[chunked_band, chunked_band]]
    profile_mix = [[1.0, 0.0] for _ in profile_bands]
    profile_is_chunked = [False] * n_len_bands + [True]
    weights = [b["weight"] for b in band_defs] + [meta.get("band_counts", {}).get("3", 0) / max(1, total)]

    # the length cutoffs to feed back into prepare_shards Config
    # (short_max, medium_max, ... = the internal edges, excluding the top which == max_seq)
    internal_cutoffs = [b["max_len"] for b in band_defs[:-1]]

    return {
        "k_elbow": k_elbow,
        "k_feasible": k_feasible,
        "k_final": n_len_bands,
        "padding_curve": curve,
        "band_edges": [b["max_len"] for b in band_defs],
        "internal_cutoffs": internal_cutoffs,      # feed these back into Config
        "band_populations": [b["count"] for b in band_defs],
        "profile_bands": profile_bands,
        "profile_mix": profile_mix,
        "profile_is_chunked": profile_is_chunked,
        "category_weights": weights,
        "total_padding_tokens": total_pad,
        "avg_shard_samples": avg_shard_samples,
        "notes": {
            "max_seq_length": max_seq,
            "num_samples": num_samples,
            "num_shards": num_shards,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="path to _shards/meta.json")
    ap.add_argument("--resident-window", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--elbow", type=float, default=0.05,
                    help="min marginal padding reduction to add a band")
    ap.add_argument("--min-band-frac", type=float, default=0.02,
                    help="merge bands holding less than this fraction of samples")
    ap.add_argument("--window-safety", type=float, default=3.0,
                    help="require per-band supply > safety * batch_size")
    ap.add_argument("--out", default=None, help="where to write band_config.json")
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    if "length_histogram" not in meta:
        raise SystemExit("meta.json has no 'length_histogram' — rebuild shards "
                         "with the histogram change in prepare_shards.py")

    cfg = derive(meta, args.resident_window, args.batch_size,
                 args.kmax, args.elbow, args.min_band_frac, args.window_safety)

    out = args.out or os.path.join(os.path.dirname(args.meta), "band_config.json")
    json.dump(cfg, open(out, "w"), indent=2)

    print(f"elbow k={cfg['k_elbow']}  window-capped k={cfg['k_feasible']}  "
          f"final k={cfg['k_final']} (+chunked)")
    print(f"band edges (max_len per band): {cfg['band_edges']}")
    print(f"internal cutoffs -> Config:    {cfg['internal_cutoffs']}")
    print(f"band populations:              {cfg['band_populations']}")
    print(f"category weights:              {[round(w,3) for w in cfg['category_weights']]}")
    print(f"padding curve (k: tokens):     "
          f"{ {k: int(v) for k,v in cfg['padding_curve'].items()} }")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()