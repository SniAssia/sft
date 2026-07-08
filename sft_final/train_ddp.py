#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ddp.py — epoch-based multi-GPU SFT with UDS selection.

Launch (Kaggle T4x2 or any N-GPU box):
    torchrun --nproc_per_node=2 train_ddp.py \
        --shards ./_shards_real --model inceptionai/jais-family-590m \
        --epochs 3 --B 8 --K 4 --lr 2e-5 --save-dir ./ckpts

Key DDP-correctness choices (why they matter):
  * steps_per_epoch is computed from the GLOBAL sample count // world_size, so
    every rank runs the SAME number of optimizer steps. If ranks did different
    step counts, the gradient all-reduce would deadlock.
  * chunked pools are DISABLED in DDP (chunked_rate=0). Otherwise one rank might
    take the "skip scoring" branch while another calls the all_gather in UDS
    select() -> collective mismatch -> hang.
  * UDS select()/commit() all_gather across ranks (global TopK + shared buffer),
    handled inside uds.py.
  * checkpoints + logging happen on rank 0 only, guarded by a barrier.
"""
from __future__ import annotations

import argparse, glob, json, math, os, time
import torch
import torch.distributed as dist

import uds_loader
from uds import UDSConfig, UDSSelector


def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--model", default="inceptionai/jais-family-590m")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--uds-warmup", type=int, default=30)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-dir", default="./ckpts")
    args = ap.parse_args()

    rank, world, local = ddp_setup()
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    is_main = rank == 0

    with open(os.path.join(args.shards, "meta.json")) as f:
        meta = json.load(f)
    vocab = int(meta["vocab_size"])

    # identical step count on every rank (see module docstring)
    per_rank = int(meta["num_samples"]) // world
    steps_per_epoch = max(1, per_rank // args.B)
    total_steps = args.epochs * steps_per_epoch
    if is_main:
        print(f"[ddp] world={world} | {meta['num_samples']} samples -> "
              f"{steps_per_epoch} steps/epoch/rank x {args.epochs} = {total_steps} steps", flush=True)

    # ---- model + DDP ----
    from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32).to(device)
    model.train(); model.config.use_cache = False
    if not args.no_grad_ckpt:
        try:    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception: model.gradient_checkpointing_enable()
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
    core = model.module if world > 1 else model

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup_frac*total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler()
    sel = UDSSelector(vocab, UDSConfig(K=args.K, alpha=1.0, svd_proj_dim=128,
                       fp_dim1=64, fp_dim2=8, start_sampling_step=args.uds_warmup, device=device))

    # ---- C++ pipeline (per-rank shard subset; infinite stream) ----
    pc = uds_loader.PipelineConfig()
    pc.shards = sorted(glob.glob(os.path.join(args.shards, "shard_*.bin")))
    pc.rank, pc.world_size, pc.seed = rank, world, args.seed
    pc.B = args.B; pc.homogeneous = True
    pc.chunked_rate = 0.0                      # DDP-safe: uniform branch across ranks
    pc.pad_id = int(meta["pad_id"]); pc.option_b_window = int(meta["max_seq_length"])
    pc.pad_to_multiple = 8; pc.num_epochs = -1
    pc.prefetch_workers = 3; pc.ring_capacity = 4
    pipe = uds_loader.DataPipeline(pc); pipe.start()

    gstep, run_loss, t0 = 0, 0.0, time.time()
    for epoch in range(args.epochs):
        if is_main: print(f"\n===== EPOCH {epoch+1}/{args.epochs} =====", flush=True)
        for _ in range(steps_per_epoch):
            pool = pipe.next_pool()
            if pool is None: break
            ii = pool.input_ids.to(device); am = pool.attention_mask.to(device); lb = pool.labels.to(device)

            if gstep < args.uds_warmup:
                t_ii, t_am, t_lb = ii, am, lb          # same branch on all ranks
            else:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = core(input_ids=ii, attention_mask=am).logits.float()
                s, z, si, se = sel.score(logits, am)
                idx = sel.select(s)                    # global all_gather TopK
                sel.commit(z[idx])                     # shared diversity buffer
                del logits
                t_ii, t_am, t_lb = ii[idx], am[idx], lb[idx]

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(input_ids=t_ii, attention_mask=t_am, labels=t_lb)
            scaler.scale(out.loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()

            run_loss += out.loss.item()
            if is_main and gstep % args.log_every == 0:
                print(f"  step {gstep:5d}/{total_steps} | loss {out.loss.item():.4f} "
                      f"| avg {run_loss/(gstep+1):.4f} | lr {sched.get_last_lr()[0]:.2e} "
                      f"| {(gstep+1)/(time.time()-t0):.2f} it/s", flush=True)
            gstep += 1

        if world > 1: dist.barrier()               # all ranks finish the epoch first
        if is_main:
            ckpt = os.path.join(args.save_dir, f"epoch{epoch+1}")
            os.makedirs(ckpt, exist_ok=True)
            core.save_pretrained(ckpt)
            print(f"  saved {ckpt}", flush=True)

    pipe.stop()
    if is_main:
        print(f"\n[ddp] DONE. formation mean {pipe.formation_mean_ms():.3f} ms | "
              f"stall mean {pipe.stall_mean_ms():.3f} ms", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()