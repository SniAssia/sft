# proxy_epoch_time.py
# Estimate full-epoch TRAIN time from a few real steps, correctly handling the
# fact that every batch has a different padded width T.
#
# Model (per batch of shape (B, T)):
#     time(B, T) = A * (B * T)  +  C * (B * T^2)
#         A  -> linear matmul cost (grows with tokens)
#         C  -> attention cost    (grows with width squared)
# Step 1 calibrate: run a FEW real timed steps at several widths -> fit A, C.
# Step 2 estimate : walk the whole epoch (loader only, no training); for each
#                   batch plug its OWN pool.padded_width into time(B,T); sum.
#
# Requires: a real model (TinyDecoder or your own), the started pipeline, torch.

import time
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# STEP 1 — calibrate: measure real step time at several widths, fit A and C.
# ---------------------------------------------------------------------------
def calibrate_A_C(model, optimizer, *, B, widths, vocab_size, device="cuda",
                  ignore_index=-100, use_amp=True, warmup=3, reps=4,
                  pad_id=0):
    """
    Runs real fwd+bwd+opt steps on synthetic batches of shape (B, T) for each T
    in `widths`, times them, and least-squares fits:
        t = A*(B*T) + C*(B*T^2)
    Returns (A, C, table) where table lists measured (T, seconds) points.
    """
    model.to(device).train()
    is_cuda = (device == "cuda" and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and is_cuda)

    def now():
        if is_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def one_step(T):
        # synthetic batch of exactly shape (B, T): timing depends on shape, not content
        input_ids = torch.randint(0, vocab_size, (B, T), device=device)
        attn = torch.ones(B, T, dtype=torch.long, device=device)
        labels = torch.randint(0, vocab_size, (B, T), device=device)
        t0 = now()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp and is_cuda):
            logits = model(input_ids, attn)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1),
                ignore_index=ignore_index)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return now() - t0

    # warmup (not timed): CUDA init / autotune on the largest width
    for _ in range(warmup):
        try:
            one_step(max(widths))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

    table = []
    for T in widths:
        samples = []
        for _ in range(reps):
            try:
                samples.append(one_step(T))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        if samples:
            table.append((T, float(np.median(samples))))
            print(f"  width T={T:5d}  ->  {1000*np.median(samples):8.2f} ms/step")

    # least-squares fit t = D + A*(B*T) + C*(B*T^2)
    # D is the FIXED per-step overhead (kernel launches, optimizer, framework)
    # that exists regardless of batch shape. Without it the fit absorbs that
    # cost by inflating A and driving C negative (attention can't be free).
    Ts = np.array([t for t, _ in table], dtype=np.float64)
    ys = np.array([s for _, s in table], dtype=np.float64)
    X = np.column_stack([np.ones_like(Ts), B * Ts, B * Ts**2])   # 1, (B*T), (B*T^2)
    coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
    D, A, C = float(coef[0]), float(coef[1]), float(coef[2])
    resid = ys - X @ coef
    print(f"\n  fit: time = {D:.4f} + {A:.3e}*(B*T) + {C:.3e}*(B*T^2)")
    print(f"       fixed overhead D = {1000*D:.1f} ms/step | "
          f"max residual = {1000*np.abs(resid).max():.2f} ms")
    if C < 0:
        print("       WARNING: C < 0 (non-physical) — add more/wider calibration points")
    return D, A, C, table


def proxy_step_seconds(batch_size, padded_width, A, C, D=0.0):
    """Estimated train time for ONE batch, from its OWN shape.
    D = fixed per-step overhead (matters a lot for small/partial batches)."""
    B, T = batch_size, padded_width
    return D + A * (B * T) + C * (B * T * T)


# ---------------------------------------------------------------------------
# STEP 2 — estimate: walk the whole epoch (no training), charge each batch its
#          own time(B, T), sum to the full-epoch estimate. Records per batch.
# ---------------------------------------------------------------------------
def estimate_epoch_seconds(pipeline, A, C, D=0.0, *, log_every=2000, idle_grace=200):
    per_batch = []
    total_s = 0.0
    padded_tokens = 0
    real_tokens = 0
    samples = 0
    idle = 0
    t_wall0 = time.perf_counter()

    pool = pipeline.next_pool()
    while True:
        if pool is None or len(pool) == 0:
            if pipeline.streamer_done():
                idle += 1
                if idle >= idle_grace:
                    break
            else:
                idle = 0
            time.sleep(0.001)
            pool = pipeline.next_pool()
            continue
        idle = 0

        B = int(pool.batch_size)
        T = int(pool.padded_width)
        est = proxy_step_seconds(B, T, A, C, D)  # this batch's own estimated time
        total_s += est

        pt = int(pool.padded_tokens); rt = int(pool.real_tokens)
        padded_tokens += pt; real_tokens += rt; samples += B
        per_batch.append({
            "batch": len(per_batch),
            "profile": int(pool.profile_index),
            "batch_size": B,
            "padded_width": T,
            "padded_tokens": pt,
            "real_tokens": rt,
            "pad_tokens": pt - rt,
            "padding_pct": 100.0 * (pt - rt) / pt if pt else 0.0,
            "est_train_seconds": est,
        })

        if log_every and len(per_batch) % log_every == 0:
            el = time.perf_counter() - t_wall0
            print(f"[{el:6.2f}s] batch={len(per_batch):6d} "
                  f"est_epoch_so_far={total_s/3600:6.3f} h "
                  f"streamed={pipeline.samples_streamed()} "
                  f"done={pipeline.streamer_done()}")
        pool = pipeline.next_pool()

    summary = {
        "batches": len(per_batch),
        "samples": samples,
        "padded_tokens": padded_tokens,
        "real_tokens": real_tokens,
        "padding_pct": 100.0 * (padded_tokens - real_tokens) / padded_tokens
                       if padded_tokens else 0.0,
        "estimated_train_seconds": total_s,
        "estimated_train_hours": total_s / 3600.0,
        "loader_walk_seconds": time.perf_counter() - t_wall0,
    }
    return summary, per_batch