# realtime_train_loop.py
# Real-time (runtime) training-time measurement over the uds_loader pipeline.
#
# The C++ side already does everything on the DATA side:
#   * ShardStreamer keeps N shards resident, fills the 4 length queues, and when
#     that window drains it loads the next N shards — automatically.
#   * BatchScheduler forms one batch (pool) at a time.
#   * Prefetcher builds the NEXT batch on background threads while you train on
#     the current one (next_pool() releases the GIL, so the GPU step and the
#     background formation truly overlap).
#
# So this file only adds the missing layer: take each formed batch and run ONE
# real, timed training step (fwd + bwd + optimizer). Timing uses CUDA events with
# synchronization — it is MEASURED at runtime, not estimated.
#
# Loop shape (exactly what you asked for):
#   [load N shards -> fill queues -> form batch] (C++, first time)
#   while batches remain:
#       t_train = REAL timed training step on this batch      # foreground, GPU
#       (meanwhile Prefetcher forms the next batch)           # background, C++
#       next_pool()  -> next batch (instant if prefetch kept up)
#   when the N-shard window empties, C++ loads the next N shards and continues.

import time
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1) A REAL model to train. Swap this for your actual model / a HF model.
#    This stub is a decoder-only transformer so the cell runs standalone and
#    the measured step time reflects real fwd+bwd compute at your chosen size.
# ---------------------------------------------------------------------------
class TinyDecoder(nn.Module):
    def __init__(self, vocab_size=32000, d_model=1024, n_layers=12,
                 n_heads=16, d_ff=4096, max_len=8192):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, activation="gelu", norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok(input_ids) + self.pos(pos)
        # causal mask + key-padding mask so padded positions cost is realistic
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        key_pad = (attention_mask == 0)          # True where pad
        x = self.blocks(x, mask=causal, src_key_padding_mask=key_pad)
        return self.head(self.norm(x))           # [B, T, vocab]


# ---------------------------------------------------------------------------
# 2) The runtime loop.  `pipeline` is your started uds_loader.DataPipeline.
# ---------------------------------------------------------------------------
def run_realtime_training(pipeline, model, optimizer, *,
                          device="cuda",
                          ignore_index=-100,
                          use_amp=True,
                          warmup_batches=3,
                          max_batches=None,
                          log_every=50,
                          idle_grace=200):
    """
    Trains on every batch the pipeline emits, timing each step for real.
    Returns a dict of MEASURED totals (no extrapolation from thin air).
    """
    model.to(device).train()
    is_cuda = (device == "cuda" and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and is_cuda)

    def now():
        if is_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    tot_train_s = 0.0      # measured GPU step time (fwd+bwd+opt)
    tot_wait_s = 0.0       # time spent waiting on next_pool (prefetch NOT hidden)
    tot_batches = 0
    tot_samples = 0
    tot_padded_tokens = 0
    tot_real_tokens = 0
    windows_seen = 0
    last_streamed = 0

    t_wall0 = time.perf_counter()
    idle = 0

    # --- prime: first next_pool() triggers the first N-shard window fill in C++
    t_get0 = time.perf_counter()
    pool = pipeline.next_pool()
    tot_wait_s += time.perf_counter() - t_get0

    while True:
        if pool is None or len(pool) == 0:
            if pipeline.streamer_done():
                idle += 1
                if idle >= idle_grace:
                    break
            else:
                idle = 0
            time.sleep(0.001)
            t_get0 = time.perf_counter()
            pool = pipeline.next_pool()
            tot_wait_s += time.perf_counter() - t_get0
            continue
        idle = 0

        # ---- move this batch to device (real tensors already collated in C++) ----
        input_ids = pool.input_ids.to(device, non_blocking=True)
        attn = pool.attention_mask.to(device, non_blocking=True)
        labels = pool.labels.to(device, non_blocking=True)
        # stub-only safety: keep ids inside the stub vocab. Delete for a real model.
        if getattr(model, "vocab_size", None):
            input_ids = input_ids.clamp_(0, model.vocab_size - 1)
            labels = torch.where(labels == ignore_index, labels,
                                 labels.clamp(0, model.vocab_size - 1))

        # ---- REAL, TIMED training step -----------------------------------------
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
        step_s = now() - t0
        # -------------------------------------------------------------------------

        # warmup steps are run but NOT counted (CUDA init / autotune / cache)
        if tot_batches >= warmup_batches:
            tot_train_s += step_s
            tot_samples += int(pool.batch_size)
            tot_padded_tokens += int(pool.padded_tokens)
            tot_real_tokens += int(pool.real_tokens)
        tot_batches += 1

        # detect an N-shard window rollover (streamed count jumps as C++ reloads)
        streamed = pipeline.samples_streamed()
        if streamed > last_streamed:
            last_streamed = streamed

        if log_every and tot_batches % log_every == 0:
            counted = max(1, tot_batches - warmup_batches)
            print(f"[{time.perf_counter()-t_wall0:7.1f}s] "
                  f"batch={tot_batches:6d} samples={tot_samples:8d} "
                  f"train={tot_train_s:7.2f}s wait={tot_wait_s:6.2f}s "
                  f"| step_ms(avg)={1000*tot_train_s/counted:6.1f} "
                  f"last_loss={loss.item():.3f} "
                  f"streamed={streamed} done={pipeline.streamer_done()}")

        if max_batches and tot_batches >= max_batches:
            break

        # ---- fetch next batch. It was formed in the BACKGROUND while we trained;
        #      wait here is the part prefetch did NOT hide (want it near zero).
        t_get0 = time.perf_counter()
        pool = pipeline.next_pool()
        tot_wait_s += time.perf_counter() - t_get0

    wall = time.perf_counter() - t_wall0
    counted = max(1, tot_batches - warmup_batches)
    tok_s = tot_padded_tokens / tot_train_s if tot_train_s > 0 else 0.0
    return {
        "batches_total": tot_batches,
        "batches_counted": counted,
        "samples_counted": tot_samples,
        "measured_train_seconds": tot_train_s,       # pure GPU step time
        "measured_wait_seconds": tot_wait_s,         # prefetch stall (not hidden)
        "wall_seconds": wall,                        # everything, end to end
        "avg_step_ms": 1000 * tot_train_s / counted,
        "throughput_tokens_per_sec": tok_s,          # MEASURED, use for any estimate
        "padded_tokens_counted": tot_padded_tokens,
        "real_tokens_counted": tot_real_tokens,
        "overlap_efficiency": tot_train_s / (tot_train_s + tot_wait_s)
                              if (tot_train_s + tot_wait_s) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# 3) Driver — paste into your notebook after building the pipeline with make().
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # These names come from YOUR notebook (uds_loader, make, meta, B, etc.):
    #   p = make(False); p.start()
    #
    # import uds_loader
    # p = make(False); p.start()
    #
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = TinyDecoder(vocab_size=int(meta.get("vocab_size", 32000)),
    #                     d_model=1024, n_layers=12, n_heads=16, d_ff=4096,
    #                     max_len=MAXLEN)
    # opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    #
    # try:
    #     stats = run_realtime_training(p, model, opt, device=device,
    #                                   ignore_index=int(meta.get("ignore_index", -100)),
    #                                   use_amp=True, warmup_batches=3, log_every=50)
    # finally:
    #     p.stop()
    #
    # import json; print(json.dumps(stats, indent=2))
    pass