# UDS SFT Pipeline

Optimized DataLoader + UDS selection for SFT, split across an **offline Python**
stage (ingest → zone-parse → tokenize → binary shards) and an **online C++**
stage (shard reading → length-aware queues → scheduler → collator → prefetch,
DDP-sharded). The model, UDS scoring, and training loop stay in Python; **libtorch
is used in C++ only to allocate/pad tensors** — every other class is hand-rolled.

```
 OFFLINE (Python)                         ONLINE (C++ ext)              ONLINE (Python)
 ┌───────────────────────┐   .bin +      ┌──────────────────────────┐  ┌───────────────────────┐
 │ prepare_shards.py     │   meta.json   │ DataPipeline             │  │ train.py              │
 │  N datasets           │──────────────▶│  ShardStreamer (mmap)    │  │  next_pool() ─┐       │
 │  source threads       │               │  4 length queues         │  │               ▼       │
 │  Tier-0 zone parse    │               │  BatchScheduler (pool B) │─▶│  UDS score (no_grad)  │
 │  case + is_chunked    │               │  Collator (libtorch pad) │  │   SVD + FastJL + Q    │
 │  Jais tokenize        │               │  Prefetcher (overlap)    │  │  TopK → train K       │
 │  shard writer         │               │  DDP shard assignment    │  │  benchmark: form/train│
 └───────────────────────┘               └──────────────────────────┘  └───────────────────────┘
```

## Layout

```
uds_pipeline/
  python/
    prepare_shards.py     # OFFLINE: datasets -> .bin shards + meta.json
    setup.py              # builds the uds_loader C++ extension
    uds.py                # Phase 12: nuclear-norm utility, FastJL, diversity buffer, TopK, DDP
    benchmark.py          # formation / training / total timing + overlap efficiency
    train.py              # online loop: pipeline -> UDS -> train, DDP-aware
    datasets.example.json
  cpp/
    include/uds/          # header-only core (torch-free except collator)
      shard_format.hpp shard_reader.hpp sample.hpp length_queues.hpp
      shard_streamer.hpp batch_scheduler.hpp distributed.hpp timer.hpp
      collator.hpp        # <-- only libtorch dependency
      prefetcher.hpp data_pipeline.hpp
    src/bindings.cpp      # pybind11 module (uds_loader)
    tests/test_core.cpp   # torch-free cross-language format test
    CMakeLists.txt
```

## 1. Offline: build shards

```bash
cd python
python prepare_shards.py \
    --config datasets.example.json \
    --out ../shards_jais590m \
    --tokenizer inceptionai/jais-family-590m \
    --max-seq-length 2048 \
    --shard-size 8192 --workers 8
```

Produces `shard_00000.bin …` + `meta.json` (pad_id, vocab, band cutoffs, etc.).
Supported input formats (Tier 0): `alpaca` (`instruction`/`input`→Context/`output`)
and `chat` (ShareGPT-style turns). `jais-family-590m` is a from-scratch base
model with no guaranteed Jinja template, so the manual `### Instruction: / ###
Input: / ### Response:` format is rendered and each zone tokenized separately to
record exact per-zone lengths.

## 2. Build the C++ extension

```bash
cd python
pip install -e .          # or: python setup.py build_ext --inplace
```

Links the libtorch shipped with your `torch` wheel, so CUDA/toolchain match
automatically. Requires a C++17 compiler.

Torch-free core test (no libtorch needed):

```bash
cd cpp && cmake -B build && cmake --build build && ./build/test_core ../shards_jais590m
```

## 3. Online: train + benchmark

Single GPU:
```bash
cd python
python train.py --shards ../shards_jais590m --model inceptionai/jais-family-590m \
    --steps 200 --B 64 --K 32 --alpha 1.0 --warmup 20
```

Multi-GPU (DDP — identical seeded shard order, per-rank subset, global TopK,
synchronized diversity buffer):
```bash
torchrun --nproc_per_node=8 train.py --shards ../shards_jais590m \
    --model inceptionai/jais-family-590m --steps 200 --B 64 --K 32
```

### Benchmark output

`train.py` prints and writes `benchmark.json` with the three headline numbers:

* **batch_formation** — the stall the loop actually sees waiting for `next_pool()`
* **training** — UDS scoring (no_grad) is timed separately; `training` is fwd+bwd+step on K
* **total (wall)** — end-to-end

plus C++-side raw formation cost and **prefetch overlap efficiency** (how much
formation was hidden behind training). Throughput is reported in samples/s and
tokens/s.

## Scheduler knobs

* `--mixed-pools` + `--w-short/--w-medium/--w-long` — compose pools across bands
  (the 30/70 knob) instead of the default length-homogeneous pools (min padding).
* `--chunked-rate` — fraction of pools drawn from the Chunked queue (Option-B
  representative-window scoring).
* `--prefetch-workers`, `--ring-capacity` — depth of the formation/training overlap.

## Known scope (per design doc)

* **Tier 1** zero-shot NLI zone detection is stubbed; Tier 0 rules only for v1.
* **Phase 13 SeCO carry-over training** is deferred. Chunked samples are *scored*
  via the Option-B window now; selected chunked samples currently train on that
  same window as a placeholder until SeCO lands. Fit samples train fully.
* `FastJL` uses fixed random projections (correct interface, JL guarantee in
  expectation); swap in a Hadamard-based FWHT for the log-factor speedup at large
  vocab.

## What's been validated

* Cross-language shard format: C++ `test_core` decodes exactly what the Python
  writer emits (band counts + homogeneous-pool invariant).
* End-to-end (model-free): shards → pipeline → padded tensors, with correct
  response-only masking, pad-to-multiple, Option-B window, and prefetch overlap.
* UDS scoring math, diversity buffer FIFO, global/local TopK, and the benchmark
  harness (synthetic logits).
* Untested here (needs Jais weights + GPU): the model forward/backward and NCCL
  DDP collectives.
