#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_shards.py
=================
OFFLINE stage of the UDS SFT data pipeline (Phases 1-4 of the design doc).

Responsibility (everything BEFORE the C++ online loader):
    N datasets  ->  source threads  ->  Ready Queue
                ->  Tier-0 zone parsing (Instruction / Context / Response)
                ->  long-zone case detection + is_chunked flag
                ->  Jais tokenization (compute prompt/context/response/total lengths)
                ->  binary shards on disk  (shard_00000.bin ...)  +  meta.json

The C++ online stage READS these shards. This file never touches the model,
UDS scoring, SVD, FastJL or training -- those live in Python's online loop.

Binary format is documented in `shard_format.md` (emitted next to this file) and
mirrored in the SHARD FORMAT section below. Keep the two in sync with the C++ reader.

Usage
-----
    python prepare_shards.py \
        --config datasets.json \
        --out ./shards_jais590m \
        --tokenizer inceptionai/jais-family-590m \
        --max-seq-length 2048 \
        --shard-size 8192 \
        --workers 8

`datasets.json` example
------------------------
[
  {"name": "alpaca_en",  "path": "tatsu-lab/alpaca",       "format": "alpaca", "split": "train", "source": "hf"},
  {"name": "sharegpt",   "path": "./data/sharegpt.jsonl",  "format": "chat",   "source": "jsonl"}
]
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------------------------------------------------------------
# SHARD FORMAT  (little-endian everywhere)  -- CONTRACT WITH THE C++ READER
# ----------------------------------------------------------------------------
# File layout of shard_XXXXX.bin:
#
#   [ HEADER            : 64 bytes                                   ]
#   [ INDEX             : num_samples * 16 bytes                     ]
#   [ DATA              : records back-to-back                       ]
#
# HEADER (64 bytes):
#   offset  size  type      field
#   0       4     char[4]   magic          = b"UDSS"
#   4       4     uint32    version        = 1
#   8       4     uint32    num_samples
#   12      4     uint32    max_seq_length
#   16      4     uint32    token_dtype    = 0 (int32 ids)
#   20      4     uint32    record_align   = 4  (bytes; records start 4-aligned)
#   24      40    ----      reserved (zero)
#
# INDEX  (one entry per sample, in shard order):
#   uint64  data_offset     # byte offset of record from START OF FILE
#   uint32  record_nbytes   # total bytes of the record
#   uint32  band            # 0=SHORT 1=MEDIUM 2=LONG 3=CHUNKED
#
# RECORD  (variable length; the three id arrays are concatenated in order
#          [prompt | context | response] so loss-mask = last response_len tokens):
#   uint32  prompt_len
#   uint32  context_len
#   uint32  response_len
#   uint32  total_len            # == prompt_len + context_len + response_len
#   uint32  is_chunked           # 0 / 1
#   uint32  case_code            # 0=FIT 1=LONG_CTX 2=LONG_RESP 3=MIXED
#   uint32  num_chunk_bounds
#   int32   prompt_ids[prompt_len]
#   int32   context_ids[context_len]
#   int32   response_ids[response_len]
#   uint32  chunk_bounds[num_chunk_bounds]   # token offsets into total seq
#
# Bands are decided here so the C++ side does O(1) queue insertion (no scan):
#   SHORT  : total_len <  band_short_max
#   MEDIUM : band_short_max <= total_len < band_medium_max
#   LONG   : band_medium_max <= total_len <= max_seq_length
#   CHUNKED: total_len > max_seq_length   (is_chunked = 1)
# ----------------------------------------------------------------------------

MAGIC = b"UDSS"
VERSION = 1
HEADER_SIZE = 64
INDEX_ENTRY_SIZE = 16  # uint64 + uint32 + uint32

BAND_SHORT, BAND_MEDIUM, BAND_LONG, BAND_CHUNKED = 0, 1, 2, 3
CASE_FIT, CASE_LONG_CTX, CASE_LONG_RESP, CASE_MIXED = 0, 1, 2, 3


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
@dataclass
class Config:
    out_dir: str
    tokenizer_name: str = "inceptionai/jais-family-590m"
    max_seq_length: int = 2048
    shard_size: int = 8192          # samples per shard
    workers: int = 8               # parse+tokenize worker threads
    queue_max: int = 20000          # backpressure on the Ready Queue

    # band cutoffs (token totals). LONG upper bound == max_seq_length.
    band_short_max: int = 512       # total < 512            -> SHORT
    band_medium_max: int = 1536     # 512 <= total < 1536    -> MEDIUM
    #                                 1536 <= total <= maxlen -> LONG
    #                                 total > maxlen          -> CHUNKED

    # response is NEVER truncated. If a sample would need response truncation
    # to fit, it is routed to CHUNKED instead. This flag only bounds the
    # representative fit-window budget used later by Option-B scoring in C++.
    option_b_window: int = 2048     # == max_seq_length by default

    add_bos: bool = True
    add_eos: bool = True            # EOS appended to the response (target end)


# ----------------------------------------------------------------------------
# SAMPLE containers
# ----------------------------------------------------------------------------
@dataclass
class RawSample:
    """Output of a source thread: raw text zones, not yet tokenized."""
    dataset: str
    instruction: str
    context: str
    response: str


@dataclass
class TokSample:
    """Fully tokenized, ready to serialize."""
    prompt_ids: List[int]
    context_ids: List[int]
    response_ids: List[int]
    is_chunked: int
    case_code: int
    chunk_bounds: List[int] = field(default_factory=list)

    @property
    def prompt_len(self) -> int: return len(self.prompt_ids)
    @property
    def context_len(self) -> int: return len(self.context_ids)
    @property
    def response_len(self) -> int: return len(self.response_ids)
    @property
    def total_len(self) -> int:
        return self.prompt_len + self.context_len + self.response_len


# ----------------------------------------------------------------------------
# JAIS TEMPLATE
# ----------------------------------------------------------------------------
# jais-family-* base models ship without a guaranteed Jinja chat_template, and
# the family's chat format is the manual "### Instruction: ... ### Response:"
# scheme. We render zones separately so we can record exact per-zone token
# lengths (tokenizing the whole string then re-splitting is fragile at
# subword boundaries). Segment-wise tokenization is standard for SFT packing.
JAIS_PROMPT_PREFIX = "### Instruction: "
JAIS_CONTEXT_PREFIX = "\n### Input: "
JAIS_RESPONSE_PREFIX = "\n### Response: "


class JaisTokenizerWrapper:
    """Thin wrapper: renders Jais zones and tokenizes each independently."""

    def __init__(self, name: str):
        from transformers import AutoTokenizer  # imported lazily
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.bos_id = self.tok.bos_token_id
        self.eos_id = self.tok.eos_token_id
        self.pad_id = self.tok.pad_token_id
        if self.pad_id is None:
            # base models often have no pad token; fall back to eos for the
            # C++ collator's pad value (recorded in meta.json).
            self.pad_id = self.eos_id
        self.vocab_size = len(self.tok)

    def _enc(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def encode_zones(self, s: RawSample, add_bos: bool, add_eos: bool
                     ) -> Tuple[List[int], List[int], List[int]]:
        # PROMPT zone = BOS + "### Instruction: <instr>"
        p = self._enc(JAIS_PROMPT_PREFIX + s.instruction)
        if add_bos and self.bos_id is not None:
            p = [self.bos_id] + p

        # CONTEXT zone (optional) = "\n### Input: <ctx>"
        c: List[int] = []
        if s.context.strip():
            c = self._enc(JAIS_CONTEXT_PREFIX + s.context)

        # RESPONSE zone = "\n### Response: <resp>" + EOS
        r = self._enc(JAIS_RESPONSE_PREFIX + s.response)
        if add_eos and self.eos_id is not None:
            r = r + [self.eos_id]

        return p, c, r


# ----------------------------------------------------------------------------
# TIER-0 ZONE PARSERS  (no ML)
# ----------------------------------------------------------------------------
class ZoneParser:
    """Registry of Tier-0 parsers. Each returns RawSample or None (skip)."""

    @staticmethod
    def parse_alpaca(rec: Dict[str, Any], dataset: str) -> Optional[RawSample]:
        # Alpaca: instruction / input(optional) / output.
        # Doc rule: Alpaca `input` field becomes CONTEXT.
        instr = (rec.get("instruction") or "").strip()
        ctx = (rec.get("input") or "").strip()
        resp = (rec.get("output") or rec.get("response") or "").strip()
        if not instr or not resp:
            return None
        return RawSample(dataset, instr, ctx, resp)

    @staticmethod
    def parse_chat(rec: Dict[str, Any], dataset: str) -> Optional[RawSample]:
        # Chat / ShareGPT: list of {role/from, content/value} turns.
        # v1 (single-turn SFT): take the last user turn as instruction and the
        # following assistant turn as response; earlier turns fold into context.
        turns = rec.get("conversations") or rec.get("messages") or []
        if not turns:
            return None

        norm = []
        for t in turns:
            role = t.get("role") or t.get("from") or ""
            content = t.get("content") or t.get("value") or ""
            role = role.lower()
            if role in ("human", "user", "prompter"):
                role = "user"
            elif role in ("gpt", "assistant", "bot", "model"):
                role = "assistant"
            elif role in ("system",):
                role = "system"
            norm.append((role, content.strip()))

        # find last assistant turn with a preceding user turn
        last_asst = None
        for i in range(len(norm) - 1, -1, -1):
            if norm[i][0] == "assistant":
                last_asst = i
                break
        if last_asst is None or last_asst == 0:
            return None

        response = norm[last_asst][1]
        # the user turn immediately before -> instruction
        instr = ""
        for j in range(last_asst - 1, -1, -1):
            if norm[j][0] == "user":
                instr = norm[j][1]
                cut = j
                break
        else:
            return None

        # everything before the instruction turn -> context (prior dialog + system)
        ctx_parts = []
        for role, content in norm[:cut]:
            if content:
                ctx_parts.append(f"{role}: {content}")
        ctx = "\n".join(ctx_parts)

        if not instr or not response:
            return None
        return RawSample(dataset, instr, ctx, response)

    @staticmethod
    def parse_pair(rec: Dict[str, Any], dataset: str,
                   prompt_field: str, response_field: str,
                   context_field: Optional[str] = None) -> Optional[RawSample]:
        # Generic single-turn: <prompt_field> -> instruction, <response_field>
        # -> response, optional <context_field> -> context. Covers aya
        # (inputs/targets) and any dataset with flat prompt/response columns.
        instr = (rec.get(prompt_field) or "").strip()
        resp = (rec.get(response_field) or "").strip()
        ctx = (rec.get(context_field) or "").strip() if context_field else ""
        if not instr or not resp:
            return None
        return RawSample(dataset, instr, ctx, resp)

    @classmethod
    def build(cls, ds: Dict[str, Any]):
        """Return a parser `rec -> Optional[RawSample]` for one dataset spec.

        Field-mapped formats read column names from the spec so different
        schemas map onto Instruction/Context/Response without new code:
            {"format": "pair", "prompt_field": "inputs", "response_field": "targets"}
        """
        name = ds["name"]
        fmt = ds["format"]
        if fmt == "alpaca":
            return lambda rec: cls.parse_alpaca(rec, name)
        if fmt == "chat":
            return lambda rec: cls.parse_chat(rec, name)
        if fmt in ("pair", "io", "fields"):
            pf = ds.get("prompt_field", "inputs")
            rf = ds.get("response_field", "targets")
            cf = ds.get("context_field")   # optional
            return lambda rec: cls.parse_pair(rec, name, pf, rf, cf)
        raise ValueError(f"Unknown format '{fmt}'. Known: alpaca, chat, pair")


# ----------------------------------------------------------------------------
# LONG-ZONE CASE DETECTION  (Phase 2, Step 2)
# ----------------------------------------------------------------------------
def decide_case(prompt_len: int, context_len: int, response_len: int,
                max_seq_length: int) -> Tuple[int, int, List[int]]:
    """
    Returns (case_code, is_chunked, chunk_bounds).

    Core rule: Response is NEVER truncated / never moved into input.
    A sample fits iff prompt+context+response <= max_seq_length.
    If it does not fit, it goes to CHUNKED and we tag WHY (long ctx / long resp /
    mixed) so C++/SeCO training later knows how to split. chunk_bounds are token
    offsets (into the concatenated [prompt|context|response] sequence) at
    max_seq_length strides -- a coarse first cut the trainer can refine.
    """
    total = prompt_len + context_len + response_len
    if total <= max_seq_length:
        return CASE_FIT, 0, []

    input_len = prompt_len + context_len
    long_ctx = input_len > max_seq_length // 2
    long_resp = response_len > max_seq_length // 2

    if long_ctx and long_resp:
        case = CASE_MIXED
    elif long_resp:
        case = CASE_LONG_RESP
    else:
        case = CASE_LONG_CTX

    # coarse contiguous chunk boundaries every max_seq_length tokens
    bounds = list(range(max_seq_length, total, max_seq_length))
    return case, 1, bounds


def band_of(total_len: int, cfg: Config, is_chunked: int) -> int:
    if is_chunked:
        return BAND_CHUNKED
    if total_len < cfg.band_short_max:
        return BAND_SHORT
    if total_len < cfg.band_medium_max:
        return BAND_MEDIUM
    return BAND_LONG


# ----------------------------------------------------------------------------
# SOURCE READERS  (Phase 1: one thread per dataset -> Ready Queue)
# ----------------------------------------------------------------------------
import glob

def iter_source(ds: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Supported sources:
        hf
        json
        jsonl
        parquet
    """

    src = ds.get("source", "jsonl")
    path = ds["path"]
    limit = ds.get("limit")

    def _capped(it):
        if limit is None:
            yield from it
        else:
            for i, rec in enumerate(it):
                if i >= limit:
                    break
                yield rec

    # -------------------------------------------------------
    # HuggingFace
    # -------------------------------------------------------
    if src == "hf":
        from datasets import load_dataset

        split = ds.get("split", "train")

        if ds.get("streaming", False):
            data = load_dataset(
                path,
                split=split,
                streaming=True
            )
        else:
            data = load_dataset(
                path,
                split=split
            )

        yield from _capped(data)

    # -------------------------------------------------------
    # Local JSONL
    # -------------------------------------------------------
    elif src == "jsonl":

        def _reader():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

        yield from _capped(_reader())

    # -------------------------------------------------------
    # Local JSON
    # -------------------------------------------------------
    elif src == "json":

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        yield from _capped(data)

    # -------------------------------------------------------
    # Local Parquet
    # -------------------------------------------------------
    elif src == "parquet":

        from datasets import load_dataset

        files = sorted(glob.glob(path))

        if len(files) == 0:
            raise FileNotFoundError(path)

        data = load_dataset(
            "parquet",
            data_files=files,
            split="train"
        )

        yield from _capped(data)

    else:
        raise ValueError(f"Unknown source '{src}'")
def source_thread(ds: Dict[str, Any], ready_q: "queue.Queue",
                  stats: Dict[str, int], stats_lock: threading.Lock) -> None:
    name = ds["name"]
    parser = ZoneParser.build(ds)
    n_read = n_kept = 0
    for rec in iter_source(ds):
        n_read += 1
        sample = parser(rec)
        if sample is not None:
            ready_q.put(sample)      # blocks on backpressure -> hides I/O latency
            n_kept += 1
    with stats_lock:
        stats["read"] = stats.get("read", 0) + n_read
        stats["kept"] = stats.get("kept", 0) + n_kept
    print(f"[source:{name}] read={n_read} kept={n_kept}", flush=True)


# ----------------------------------------------------------------------------
# SHARD WRITER  (Phase 4)  -- single writer thread, thread-safe via its queue
# ----------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.buf: List[TokSample] = []
        self.shard_idx = 0
        self.total_written = 0
        self.band_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    def add(self, ts: TokSample) -> None:
        self.buf.append(ts)
        if len(self.buf) >= self.cfg.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        path = os.path.join(self.cfg.out_dir, f"shard_{self.shard_idx:05d}.bin")
        self._write_shard(path, self.buf)
        self.total_written += len(self.buf)
        print(f"[shard] wrote {path}  ({len(self.buf)} samples)", flush=True)
        self.buf = []
        self.shard_idx += 1

    def _record_bytes(self, ts: TokSample) -> Tuple[bytes, int]:
        header = struct.pack(
            "<7I",
            ts.prompt_len, ts.context_len, ts.response_len, ts.total_len,
            ts.is_chunked, ts.case_code, len(ts.chunk_bounds),
        )
        ids = ts.prompt_ids + ts.context_ids + ts.response_ids
        body = struct.pack(f"<{len(ids)}i", *ids) if ids else b""
        bounds = (struct.pack(f"<{len(ts.chunk_bounds)}I", *ts.chunk_bounds)
                  if ts.chunk_bounds else b"")
        rec = header + body + bounds
        return rec, len(rec)

    def _write_shard(self, path: str, samples: List[TokSample]) -> None:
        cfg = self.cfg
        n = len(samples)

        # Build records + compute offsets. Data starts after header+index.
        data_start = HEADER_SIZE + n * INDEX_ENTRY_SIZE
        records: List[bytes] = []
        index: List[Tuple[int, int, int]] = []  # (offset, nbytes, band)
        cursor = data_start
        for ts in samples:
            rec, nbytes = self._record_bytes(ts)
            band = band_of(ts.total_len, cfg, ts.is_chunked)
            self.band_counts[band] += 1
            index.append((cursor, nbytes, band))
            records.append(rec)
            cursor += nbytes

        with open(path, "wb") as f:
            # header
            f.write(MAGIC)
            f.write(struct.pack("<5I", VERSION, n, cfg.max_seq_length, 0, 4))
            f.write(b"\x00" * (HEADER_SIZE - 24))
            # index
            for off, nb, band in index:
                f.write(struct.pack("<QII", off, nb, band))
            # data
            for rec in records:
                f.write(rec)


# ----------------------------------------------------------------------------
# WORKER  (Phase 2+3: parse-zone -> case -> tokenize)  Ready Q -> Writer Q
# ----------------------------------------------------------------------------
def worker_thread(tokenizer: JaisTokenizerWrapper, cfg: Config,
                  ready_q: "queue.Queue", writer_q: "queue.Queue",
                  sentinel: object) -> None:
    while True:
        item = ready_q.get()
        if item is sentinel:
            ready_q.put(sentinel)   # propagate to sibling workers
            break
        s: RawSample = item
        p, c, r = tokenizer.encode_zones(s, cfg.add_bos, cfg.add_eos)
        case, is_chunked, bounds = decide_case(
            len(p), len(c), len(r), cfg.max_seq_length)
        ts = TokSample(prompt_ids=p, context_ids=c, response_ids=r,
                       is_chunked=is_chunked, case_code=case, chunk_bounds=bounds)
        writer_q.put(ts)


# ----------------------------------------------------------------------------
# ORCHESTRATION
# ----------------------------------------------------------------------------
def run(cfg: Config, datasets: List[Dict[str, Any]]) -> None:
    print(f"[init] loading tokenizer {cfg.tokenizer_name} ...", flush=True)
    tokenizer = JaisTokenizerWrapper(cfg.tokenizer_name)
    print(f"[init] vocab={tokenizer.vocab_size} "
          f"bos={tokenizer.bos_id} eos={tokenizer.eos_id} pad={tokenizer.pad_id}",
          flush=True)

    sentinel = object()
    ready_q: "queue.Queue" = queue.Queue(maxsize=cfg.queue_max)
    writer_q: "queue.Queue" = queue.Queue(maxsize=cfg.queue_max)
    stats: Dict[str, int] = {}
    stats_lock = threading.Lock()

    # source threads
    sources = [threading.Thread(target=source_thread,
                                args=(ds, ready_q, stats, stats_lock),
                                name=f"src-{ds['name']}", daemon=True)
               for ds in datasets]

    # worker threads
    workers = [threading.Thread(target=worker_thread,
                                args=(tokenizer, cfg, ready_q, writer_q, sentinel),
                                name=f"wrk-{i}", daemon=True)
               for i in range(cfg.workers)]

    # writer thread (single, owns file handles)
    writer = ShardWriter(cfg)
    writer_done = threading.Event()

    def writer_loop():
        while True:
            item = writer_q.get()
            if item is sentinel:
                break
            writer.add(item)
        writer.flush()
        writer_done.set()

    wt = threading.Thread(target=writer_loop, name="writer", daemon=True)

    t0 = time.time()
    for s in sources: s.start()
    for w in workers: w.start()
    wt.start()

    # wait for all source threads, then signal workers to drain
    for s in sources: s.join()
    ready_q.put(sentinel)
    for w in workers: w.join()
    writer_q.put(sentinel)
    writer_done.wait()

    dt = time.time() - t0
    _write_meta(cfg, tokenizer, writer, stats, dt)
    print(f"[done] {writer.total_written} samples in {writer.shard_idx} shards "
          f"({dt:.1f}s). bands={writer.band_counts}", flush=True)


def _write_meta(cfg: Config, tokenizer: JaisTokenizerWrapper,
                writer: ShardWriter, stats: Dict[str, int], dt: float) -> None:
    meta = {
        "format_version": VERSION,
        "magic": MAGIC.decode(),
        "tokenizer": cfg.tokenizer_name,
        "vocab_size": tokenizer.vocab_size,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "pad_id": tokenizer.pad_id,          # C++ collator pads with this
        "max_seq_length": cfg.max_seq_length,
        "bands": {
            "short_max": cfg.band_short_max,
            "medium_max": cfg.band_medium_max,
            "long_max": cfg.max_seq_length,
        },
        "option_b_window": cfg.option_b_window,
        "num_shards": writer.shard_idx,
        "num_samples": writer.total_written,
        "band_counts": writer.band_counts,
        "case_codes": {"FIT": 0, "LONG_CTX": 1, "LONG_RESP": 2, "MIXED": 3},
        "band_codes": {"SHORT": 0, "MEDIUM": 1, "LONG": 2, "CHUNKED": 3},
        "loss_mask_rule": "loss on last response_len tokens only",
        "zone_order_in_record": ["prompt", "context", "response"],
        "source_stats": stats,
        "build_seconds": round(dt, 1),
    }
    with open(os.path.join(cfg.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[meta] wrote {os.path.join(cfg.out_dir, 'meta.json')}", flush=True)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="UDS SFT offline shard builder")
    ap.add_argument("--config", required=True, help="datasets JSON spec")
    ap.add_argument("--out", required=True, help="output shard dir")
    ap.add_argument("--tokenizer", default="inceptionai/jais-family-590m")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--shard-size", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--band-short-max", type=int, default=512)
    ap.add_argument("--band-medium-max", type=int, default=1536)
    ap.add_argument("--no-bos", action="store_true")
    ap.add_argument("--no-eos", action="store_true")
    args = ap.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        datasets = json.load(f)

    cfg = Config(
        out_dir=args.out,
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length,
        shard_size=args.shard_size,
        workers=args.workers,
        band_short_max=args.band_short_max,
        band_medium_max=args.band_medium_max,
        option_b_window=args.max_seq_length,
        add_bos=not args.no_bos,
        add_eos=not args.no_eos,
    )
    run(cfg, datasets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
