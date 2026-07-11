#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_proxy.py — run the PROXY-time / padding benchmark over one epoch, comparing
the round-robin category scheduler ("ours") against length-agnostic baseline.

NO REAL TRAINING happens. In the distributed path each rank runs the proxy over
its own shard subset and the totals are all-reduced — i.e. the distributed system
still runs, it just does the proxy instead of a real optimizer step.

Single process:
    python run_proxy.py --shards ./_shards --B 32 --window 4

Distributed (still runs the DDP machinery, proxy instead of training):
    torchrun --nproc_per_node=2 run_proxy.py --shards ./_shards --B 32 --window 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import uds_loader
from proxy_benchmark import run_epoch_proxy, compare, ProxyModel, EpochStats

# CAT1 Short40/Medium60 · CAT2 Medium45/Long55 · CAT3 Chunked
DEFAULT_BANDS = [[0, 1], [1, 2], [3, 3]]
DEFAULT_MIX = [[0.40, 0.60], [0.45, 0.55], [1.0, 0.0]]
DEFAULT_CHUNK = [False, False, True]


def build_pipeline(args, meta, rank, world, baseline: bool):
    pc = uds_loader.PipelineConfig()
    pc.shards = sorted(glob.glob(os.path.join(args.shards, "shard_*.bin")))
    pc.rank, pc.world_size, pc.seed = rank, world, args.seed
    pc.num_epochs = 1                       # exactly one epoch of proxy
    pc.max_queue_occupancy = args.occupancy
    pc.resident_window = args.window
    pc.B = args.B
    pc.baseline = baseline
    if not baseline:
        pc.profile_bands = DEFAULT_BANDS
        pc.profile_mix = DEFAULT_MIX
        pc.profile_is_chunked = DEFAULT_CHUNK
    pc.pad_id = int(meta["pad_id"])
    pc.option_b_window = int(meta.get("option_b_window", meta["max_seq_length"]))
    pc.pad_to_multiple = args.pad_to_multiple
    pc.prefetch_workers = args.prefetch_workers
    pc.ring_capacity = args.ring_capacity
    p = uds_loader.DataPipeline(pc)
    return p


def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch, torch.distributed as dist
        dist.init_process_group(backend="gloo")   # gloo: works CPU+GPU, no real tensors on GPU here
        return dist.get_rank(), dist.get_world_size(), dist
    return 0, 1, None


def all_reduce_stats(st: EpochStats, dist) -> EpochStats:
    if dist is None:
        return st
    import torch
    vec = torch.tensor([st.pools, st.samples, st.padded_tokens, st.real_tokens,
                        st.pad_tokens, st.proxy_time, st.fallback_pools,
                        st.skipped_categories, st.samples_streamed], dtype=torch.float64)
    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    (st.pools, st.samples, st.padded_tokens, st.real_tokens, st.pad_tokens,
     st.proxy_time, st.fallback_pools, st.skipped_categories,
     st.samples_streamed) = (int(vec[0]), int(vec[1]), int(vec[2]), int(vec[3]),
                             int(vec[4]), float(vec[5]), int(vec[6]),
                             int(vec[7]), int(vec[8]))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--window", type=int, default=4, help="resident shard window W")
    ap.add_argument("--occupancy", type=int, default=50000)
    ap.add_argument("--pad-to-multiple", type=int, default=8)
    ap.add_argument("--prefetch-workers", type=int, default=3)
    ap.add_argument("--ring-capacity", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    # proxy cost model (alpha=1,beta=0 => relative padded-token units)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--out", default="proxy_benchmark.json")
    args = ap.parse_args()

    rank, world, dist = ddp_setup()
    is_main = rank == 0
    meta = json.load(open(os.path.join(args.shards, "meta.json")))
    model = ProxyModel(alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    # ---- ours (round-robin categories) ----
    p_ours = build_pipeline(args, meta, rank, world, baseline=False)
    p_ours.start()
    ours = run_epoch_proxy(
    p_ours,
    model,
    method="round_robin",
    )

    p_ours.stop()

    # ---- baseline (length-agnostic random batching) ----
    p_base = build_pipeline(args, meta, rank, world, baseline=True)
    p_base.start()
    base = run_epoch_proxy(
    p_base,
    model,
    method="baseline",
    )
    p_base.stop()

    ours = all_reduce_stats(ours, dist)
    base = all_reduce_stats(base, dist)

    if is_main:
        print(compare(ours, base))
        json.dump({"ours": ours.summary(), "baseline": base.summary(),
                   "world_size": world, "B": args.B, "resident_window": args.window,
                   "proxy_model": {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma}},
                  open(args.out, "w"), indent=2)
        print(f"\n[proxy] wrote {args.out}")

    if dist is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
