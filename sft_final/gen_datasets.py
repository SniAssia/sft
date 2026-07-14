#!/usr/bin/env python3
"""
gen_datasets.py — generate 3 offline instruction/response JSONL datasets with
realistic text and controlled length distributions, so the length-band pipeline
has data with genuine short/medium/long spread (no network / HF needed).

Each record: {"instruction": ..., "input": ..., "output": ...}  (alpaca format)

Datasets:
  ds_small_60k.jsonl   ~60k  short-skewed  (fills SHORT, some MEDIUM)
  ds_tiny_20k.jsonl    ~20k  medium-skewed (fills MEDIUM, some SHORT/LONG)
  ds_big_200k.jsonl    ~200k full spread   (SHORT..LONG..CHUNKED)
"""
import json, random, os

random.seed(1234)
OUT = os.path.dirname(os.path.abspath(__file__))

# a small vocabulary of real words so tokenization behaves like real text
WORDS = ("the quick brown fox jumps over lazy dog while data flows through "
         "pipelines and models learn patterns from tokens across many batches "
         "with attention layers computing weighted sums of hidden states during "
         "training steps that optimize loss over gradients and parameters").split()
INSTRUCTIONS = [
    "Explain the following concept in detail:",
    "Summarize the text below:",
    "Answer the question:",
    "Translate the passage:",
    "Write a short analysis of:",
    "Continue the following story:",
    "Describe step by step how to:",
]

def make_text(n_words):
    return " ".join(random.choice(WORDS) for _ in range(max(1, n_words)))

def emit(path, n, length_choices):
    """length_choices: list of (instr_words, output_words) tuples to sample."""
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            iw, ow = random.choice(length_choices)
            rec = {
                "instruction": random.choice(INSTRUCTIONS) + " " + make_text(iw),
                "input": "",
                "output": make_text(ow),
            }
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {path}  ({n} samples)")

# word counts chosen so that instruction+output token totals land across bands.
# (~1.3 tokens/word for this vocab; totals below are word-sums.)
SHORT   = (10, 30)      # ~ short
MEDIUM  = (60, 180)     # ~ medium
LONG    = (250, 500)    # ~ long
XLONG   = (700, 1100)   # ~ chunked (> max_seq_len at 1024)

# ds_small_60k: mostly short, some medium
emit(os.path.join(OUT, "ds_small_60k.jsonl"), 60000,
     [SHORT]*7 + [MEDIUM]*3)

# ds_tiny_20k: mostly medium, some short + long
emit(os.path.join(OUT, "ds_tiny_20k.jsonl"), 20000,
     [SHORT]*2 + [MEDIUM]*5 + [LONG]*3)

# ds_big_200k: full spread including long/chunked
emit(os.path.join(OUT, "ds_big_200k.jsonl"), 200000,
     [SHORT]*4 + [MEDIUM]*3 + [LONG]*2 + [XLONG]*1)

print("done")