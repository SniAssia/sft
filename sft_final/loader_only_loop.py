# loader_only_loop.py
# Run the FULL pipeline (resident shards -> 4 queues -> form batch -> prefetch
# next) over the whole epoch, with the training step REPLACED by "get the next
# batch". Measures pure data-pipeline throughput: how fast batches can be formed
# back-to-back, which is the ceiling on how fast a GPU could ever be fed.
#
# Two modes:
#   sim_step_s = 0.0  -> max loader throughput (consumer is instant). This is the
#                        ceiling; overlap is irrelevant because there's nothing to
#                        hide formation behind.
#   sim_step_s > 0.0  -> pretend each "training step" takes this many seconds
#                        (a stand-in for the GPU). Now the background prefetcher
#                        overlaps formation with that delay, and measured_wait_s
#                        tells you whether the loader keeps up (wait≈0 = yes).

import time


def run_loader_only(pipeline, *, sim_step_s=0.0, log_every=2000, idle_grace=200):
    """
    Iterate the whole epoch replacing training with a get-next-batch (plus an
    optional fixed sim_step_s stand-in for the GPU). Records per-batch padding.
    Returns (summary_dict, per_batch_list).
    """
    per_batch = []
    batches = 0
    samples = 0
    padded_tokens = 0
    real_tokens = 0
    wait_s = 0.0        # time blocked in next_pool (formation NOT hidden)
    sim_s = 0.0         # time spent in the fake "step"
    idle = 0

    t_wall0 = time.perf_counter()

    # prime: first next_pool() triggers the first N-shard window fill in C++
    t0 = time.perf_counter()
    pool = pipeline.next_pool()
    wait_s += time.perf_counter() - t0

    while True:
        if pool is None or len(pool) == 0:
            if pipeline.streamer_done():
                idle += 1
                if idle >= idle_grace:
                    break
            else:
                idle = 0
            time.sleep(0.001)
            t0 = time.perf_counter()
            pool = pipeline.next_pool()
            wait_s += time.perf_counter() - t0
            continue
        idle = 0

        # ---- this is where a real training step would go. Replaced by nothing,
        #      or a fixed stand-in delay if sim_step_s > 0. ----
        pt = int(pool.padded_tokens)
        rt = int(pool.real_tokens)
        per_batch.append({
            "batch": batches,
            "profile": int(pool.profile_index),
            "batch_size": int(pool.batch_size),
            "padded_width": int(pool.padded_width),
            "padded_tokens": pt,
            "real_tokens": rt,
            "pad_tokens": pt - rt,
            "padding_pct": 100.0 * (pt - rt) / pt if pt else 0.0,
            "fell_back": bool(pool.fell_back),
        })
        batches += 1
        samples += int(pool.batch_size)
        padded_tokens += pt
        real_tokens += rt

        if sim_step_s > 0.0:
            t0 = time.perf_counter()
            time.sleep(sim_step_s)          # stand-in for the GPU step
            sim_s += time.perf_counter() - t0

        if log_every and batches % log_every == 0:
            el = time.perf_counter() - t_wall0
            print(f"[{el:7.2f}s] batch={batches:6d} samples={samples:8d} "
                  f"wait={wait_s:6.3f}s sim={sim_s:6.2f}s "
                  f"| batches/s={batches/el:8.1f} "
                  f"streamed={pipeline.samples_streamed()} "
                  f"done={pipeline.streamer_done()}")

        # ---- immediately go get the next batch (formed in background meanwhile) ----
        t0 = time.perf_counter()
        pool = pipeline.next_pool()
        wait_s += time.perf_counter() - t0

    wall = time.perf_counter() - t_wall0
    form_wall = wall - sim_s                 # loader time excluding the fake step
    summary = {
        "batches": batches,
        "samples": samples,
        "padded_tokens": padded_tokens,
        "real_tokens": real_tokens,
        "pad_tokens": padded_tokens - real_tokens,
        "padding_pct": 100.0 * (padded_tokens - real_tokens) / padded_tokens
                       if padded_tokens else 0.0,
        "wall_seconds": wall,
        "sim_step_seconds_total": sim_s,
        "loader_wall_seconds": form_wall,            # wall minus the fake steps
        "measured_wait_seconds": wait_s,             # formation NOT hidden
        "batches_per_sec": batches / form_wall if form_wall > 0 else 0.0,
        "samples_per_sec": samples / form_wall if form_wall > 0 else 0.0,
        "formation_total_s": float(pipeline.formation_total_s()),
        "stall_total_s": float(pipeline.stall_total_s()),
        "fallback_pools": int(pipeline.fallback_pools()),
        "skipped_categories": int(pipeline.skipped_categories()),
        "samples_streamed": int(pipeline.samples_streamed()),
    }
    return summary, per_batch