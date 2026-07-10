# Length-Aware Category Batching + Proxy Benchmark (SFT data pipeline)

A hand-written C++ data engine (libtorch only for tensors) that batches
pre-tokenized SFT shards by **length category**, and a **proxy benchmark** that
estimates one-epoch training time *without real training* and compares against a
length-agnostic baseline.

## Categories (round-robin, one per batch)
- **CAT1** = Short 40% / Medium 60%
- **CAT2** = Medium 45% / Long 55%
- **CAT3** = Chunked (100%)

Batches rotate CAT1 → CAT2 → CAT3 → … . Within a category the two bands are filled
at the configured split. If a band's queue is empty:
- **not exhausted** (resident shards still hold that band) → fill the rest from the
  other band now, and **signal a background refill** (next time the category uses
  the configured split again);
- **exhausted** (resident shards fully passed for that band) → fill the rest from
  the other band, no signal.
If both bands of a category are dry → **skip to the next category**; if all
categories are dry → wait for the next resident **window** of shards; if all shards
are passed → the epoch ends.

`resident_window` (W) = how many shards are held resident together (RAM vs smoother
mixing / fewer fallbacks).

## Proxy training time (no real training)
The GPU processes every token in the padded tensor, so step time ∝ padded tokens
`B·T_padded`. One-epoch proxy time = Σ `alpha·(B_i·T_padded_i) + beta`. With
`alpha=1, beta=0` this is the total padded-token count — a relative measure for
comparing batching methods (fit alpha/beta from a few timed steps for seconds).

## Benchmark metrics (ours vs baseline)
padding %, padded tokens, proxy train time, useful-token ratio, per-category
padding, fallback pools, skipped categories, empty-queue alerts, formation/stall.

## Build
```
python setup.py build_ext --inplace     # builds uds_loader*.so
```
Torch-free core test:
```
g++ -std=c++17 -O2 -I . test_core.cpp -o test_core -lpthread && ./test_core ./_shards
```

## Run the proxy benchmark
Build shards (real data):
```
python prepare_shards.py --config datasets_real.json --out ./_shards \
    --tokenizer facebook/opt-125m --max-seq-length 1024 --shard-size 1024 --workers 4
```
Single process (ours vs baseline):
```
python run_proxy.py --shards ./_shards --B 32 --window 4
```
Distributed (DDP runs for real, does the proxy instead of training):
```
torchrun --nproc_per_node=2 run_proxy.py --shards ./_shards --B 32 --window 4
```
Or open **proxy_benchmark.ipynb** (Colab T4×2) and run top to bottom.

## Files
- **C++ engine:** `length_queues.hpp` (queues + hungry/exhausted flags),
  `shard_streamer.hpp` (resident-window streaming + exhaustion),
  `batch_scheduler.hpp` (round-robin categories + fallback + baseline),
  `collator.hpp` (padding + proxy token counts), `prefetcher.hpp`,
  `data_pipeline.hpp`, `bindings.cpp`, `shard_reader.hpp`, `shard_format.hpp`,
  `sample.hpp`, `distributed.hpp`, `timer.hpp`, `test_core.cpp`.
- **Offline:** `prepare_shards.py`, `datasets_real.json`.
- **Proxy benchmark:** `proxy_benchmark.py` (metrics), `run_proxy.py` (single/DDP),
  `benchmark.py` (timing helpers), `proxy_benchmark.ipynb`.
- **UDS (separate, NOT used here):** `uds.py`, `train.py`, `train_ddp.py`,
  `test_pipeline.ipynb`.

## Build shards format / tokenizer
`prepare_shards.py` supports `alpaca`, `chat`, and `pair` (e.g. aya inputs/targets)
formats via `datasets_real.json`. Any HF tokenizer via `--tokenizer`.
