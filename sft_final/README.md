# sft_final — complete flat bundle

Drop-in replacement for the `sft_final/` folder. Fixes the earlier repo state
(6 missing headers + flat-include breakage). Everything is flat; includes are
same-directory; `setup.py` builds from this folder.

## Build
    python setup.py build_ext --inplace      # creates uds_loader*.so here

## Test (order: batching -> training)
Open `test_pipeline.ipynb` and run top to bottom, or from a terminal:

    # 1) build shards (offline)
    python prepare_shards.py --config datasets.example.json --out ./shards \
        --tokenizer inceptionai/jais-family-590m --max-seq-length 2048

    # 2) train + benchmark (single GPU)
    python train.py --shards ./shards --model inceptionai/jais-family-590m \
        --steps 200 --B 64 --K 32 --warmup 20

    # torch-free core test
    g++ -std=c++17 -O2 -I . test_core.cpp -o test_core -lpthread && ./test_core ./shards

## Files
  headers:  shard_format sample length_queues shard_streamer distributed timer
            shard_reader batch_scheduler collator prefetcher data_pipeline  (.hpp)
  cpp:      bindings.cpp (pybind), test_core.cpp
  python:   prepare_shards.py uds.py benchmark.py train.py setup.py
  notebook: test_pipeline.ipynb
