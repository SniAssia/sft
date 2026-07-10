#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — online SFT loop.

Wires the C++ `uds_loader.DataPipeline` (shard reading -> length queues ->
scheduler -> collator -> prefetch, DDP-sharded) to the Python-owned model +
UDS selection (Phase 12) + training, and measures batch-formation / training /
total time via benchmark.Benchmark.

Single GPU:
    python train.py --shards ./shards_jais590m --model inceptionai/jais-family-590m \
        --steps 200 --B 64 --K 32

Multi-GPU (DDP):
    torchrun --nproc_per_node=8 train.py --shards ./shards_jais590m \
        --model inceptionai/jais-family-590m --steps 200 --B 64 --K 32
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import torch
import torch.distributed as dist

import uds_loader  # the compiled C++ extension (build via python/setup.py)
from uds import UDSConfig, UDSSelector
from benchmark import Benchmark


# ----------------------------------------------------------------------------
def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def load_meta(shard_dir: str) -> dict:
    with open(os.path.join(shard_dir, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def build_pipeline_config(args, meta, rank, world) -> "uds_loader.PipelineConfig":
    cfg = uds_loader.PipelineConfig()
    cfg.shards = sorted(glob.glob(os.path.join(args.shards, "shard_*.bin")))
    cfg.rank = rank
    cfg.world_size = world
    cfg.seed = args.seed
    cfg.num_epochs = args.epochs
    cfg.B = args.B
    cfg.homogeneous = not args.mixed_pools
    cfg.fit_band_weights = [args.w_short, args.w_medium, args.w_long]
    cfg.chunked_rate = args.chunked_rate
    cfg.pad_id = int(meta["pad_id"])
    cfg.ignore_index = -100
    cfg.option_b_window = int(meta.get("option_b_window", meta["max_seq_length"]))
    cfg.pad_to_multiple = args.pad_to_multiple
    cfg.prefetch_workers = args.prefetch_workers
    cfg.ring_capacity = args.ring_capacity
    return cfg


def load_model(name: str, device: str):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, torch_dtype=torch.float32)
    model.to(device)
    model.train()
    return model


def cpp_stats(pipeline) -> dict:
    return {
        "formation_total_s": pipeline.formation_total_s(),
        "formation_mean_ms": pipeline.formation_mean_ms(),
        "formation_count": pipeline.formation_count(),
        "stall_total_s": pipeline.stall_total_s(),
        "stall_mean_ms": pipeline.stall_mean_ms(),
        "samples_streamed": pipeline.samples_streamed(),
    }


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--model", default="inceptionai/jais-family-590m")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lr", type=float, default=1e-5)
    # pool / selection
    ap.add_argument("--B", type=int, default=64, help="candidate pool size")
    ap.add_argument("--K", type=int, default=32, help="selected per pool")
    ap.add_argument("--alpha", type=float, default=1.0, help="diversity weight")
    ap.add_argument("--warmup", type=int, default=20, help="steps of full-pool training")
    ap.add_argument("--svd-proj-dim", type=int, default=256)
    # scheduler
    ap.add_argument("--mixed-pools", action="store_true")
    ap.add_argument("--w-short", type=float, default=1.0)
    ap.add_argument("--w-medium", type=float, default=1.0)
    ap.add_argument("--w-long", type=float, default=1.0)
    ap.add_argument("--chunked-rate", type=float, default=0.1)
    ap.add_argument("--pad-to-multiple", type=int, default=8)
    # prefetch
    ap.add_argument("--prefetch-workers", type=int, default=3)
    ap.add_argument("--ring-capacity", type=int, default=4)
    ap.add_argument("--out", default="benchmark.json")
    args = ap.parse_args()

    rank, world, local = ddp_setup()
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    is_main = rank == 0

    meta = load_meta(args.shards)
    vocab = int(meta["vocab_size"])

    # ---- model ----
    model = load_model(args.model, device)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---- UDS ----
    ucfg = UDSConfig(K=args.K, alpha=args.alpha, svd_proj_dim=args.svd_proj_dim,
                     start_sampling_step=args.warmup, device=device, seed=args.seed)
    selector = UDSSelector(vocab, ucfg)

    # ---- C++ pipeline ----
    pcfg = build_pipeline_config(args, meta, rank, world)
    pipeline = uds_loader.DataPipeline(pcfg)
    pipeline.start()

    bench = Benchmark(sync=lambda: torch.cuda.synchronize() if torch.cuda.is_available() else None)
    bench.start()

    core = model.module if world > 1 else model

    for step in range(args.steps):
        # ---- batch formation (stall the loop actually sees) ----
        with bench.phase("batch_formation"):
            pool = pipeline.next_pool()
        if pool is None:
            break

        input_ids = pool.input_ids.to(device, non_blocking=True)
        attn = pool.attention_mask.to(device, non_blocking=True)
        labels = pool.labels.to(device, non_blocking=True)

        warming = step < args.warmup
        if warming:
            train_ids, train_attn, train_labels = input_ids, attn, labels
        else:
            # ---- UDS scoring (no_grad) : selection ----
            with bench.phase("uds_scoring", gpu=True):
                with torch.no_grad():
                    logits = core(input_ids=input_ids, attention_mask=attn).logits
                s_total, z, s_intra, s_inter = selector.score(logits, attn)
                idx = selector.select(s_total)
                selector.commit(z[idx])
            train_ids = input_ids[idx]
            train_attn = attn[idx]
            train_labels = labels[idx]

        # ---- training forward + backward + step ----
        with bench.phase("training", gpu=True):
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=train_ids, attention_mask=train_attn, labels=train_labels)
            out.loss.backward()
            opt.step()

        bench.add_samples(train_ids.shape[0], tokens=int(train_attn.sum().item()))

        if is_main and step % 10 == 0:
            print(f"[step {step:4d}] pool={len(pool)} train={train_ids.shape[0]} "
                  f"loss={out.loss.item():.4f} "
                  f"chunked={'Y' if pool.is_chunked else 'n'} "
                  f"qsize(S/M/L/C)={pipeline.queue_size(0)}/{pipeline.queue_size(1)}/"
                  f"{pipeline.queue_size(2)}/{pipeline.queue_size(3)}", flush=True)

    bench.stop()
    pipeline.stop()

    if is_main:
        stats = cpp_stats(pipeline)
        print(bench.report(stats))
        bench.save(args.out, stats)
        print(f"[bench] wrote {args.out}")

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
