#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uds.py — Phase 12 UDS selection layer (Python side; owns the model).

Given a candidate pool of B collated samples, run ONE no_grad forward to get
logits, then score:

    s_intra  = || L ||_*            nuclear norm (batched SVD)   -- utility
    z        = FastJL(L)            compact fingerprint          -- for diversity
    s_inter  = mean L2 to buffer Q  diversity term
    s_total  = s_intra + alpha * s_inter
    TopK -> K indices

Then update the FIFO diversity buffer Q with the selected fingerprints.

Cost control: nuclear norm on raw [T, vocab] logits is huge, so we right-project
logits to `svd_proj_dim` with a FIXED random matrix before the SVD (preserves
relative singular-value structure well enough for ranking). FastJL likewise maps
V -> fp_dim1 -> fp_dim2 with fixed projections, so buffer entries are tiny
(fp_dim2), keeping the DDP buffer sync cheap ("tiny-vector syncs").

DDP: scores are all-gathered so TopK is GLOBAL; each rank then trains the
selected samples it locally owns. The buffer is synchronized so every rank keeps
one shared diversity history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class UDSConfig:
    K: int = 32                     # selected per pool (K < B)
    alpha: float = 1.0              # diversity weight
    buffer_capacity: int = 1024     # FIFO diversity buffer Q
    svd_proj_dim: int = 256         # logits projected to this before SVD
    fp_dim1: int = 128              # FastJL stage 1
    fp_dim2: int = 8                # FastJL stage 2 (buffer entry dim)
    start_sampling_step: int = 100  # warm-up: train full pool before this
    seed: int = 12345
    device: str = "cuda"
    dtype: torch.dtype = torch.float32


class FastJL:
    """Fixed two-stage random projection V -> fp_dim1 -> fp_dim2.

    A practical stand-in for the Hadamard-based FastJL transform: fixed random
    matrices give the same interface and JL distance-preservation in expectation.
    Swap in a true FWHT here if you want the log-factor speedup at large V.
    """

    def __init__(self, vocab: int, cfg: UDSConfig):
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        scale1 = 1.0 / math.sqrt(cfg.fp_dim1)
        scale2 = 1.0 / math.sqrt(cfg.fp_dim2)
        self.G1 = (torch.randn(vocab, cfg.fp_dim1, generator=g) * scale1).to(
            cfg.device, cfg.dtype)
        self.G2 = (torch.randn(cfg.fp_dim1, cfg.fp_dim2, generator=g) * scale2).to(
            cfg.device, cfg.dtype)

    @torch.no_grad()
    def __call__(self, summary: torch.Tensor) -> torch.Tensor:
        # summary: [P, V] -> [P, fp_dim2]
        return (summary @ self.G1) @ self.G2


class DiversityBuffer:
    """FIFO buffer Q of fingerprints; diversity = mean L2 distance to contents."""

    def __init__(self, cfg: UDSConfig):
        self.cap = cfg.buffer_capacity
        self.dim = cfg.fp_dim2
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.buf = torch.empty((0, self.dim), device=cfg.device, dtype=cfg.dtype)

    @torch.no_grad()
    def mean_l2(self, z: torch.Tensor) -> torch.Tensor:
        # z: [P, dim] -> [P] mean L2 distance to every buffer entry
        if self.buf.shape[0] == 0:
            return torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        d = torch.cdist(z, self.buf)          # [P, |Q|]
        return d.mean(dim=1)

    @torch.no_grad()
    def add(self, z: torch.Tensor) -> None:
        self.buf = torch.cat([self.buf, z.to(self.device, self.dtype)], dim=0)
        if self.buf.shape[0] > self.cap:
            self.buf = self.buf[-self.cap:]

    @torch.no_grad()
    def sync_ddp(self) -> None:
        """Merge every rank's newest additions into one shared history."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        world = dist.get_world_size()
        # gather variable-sized buffers by padding to the max size
        n = torch.tensor([self.buf.shape[0]], device=self.device)
        sizes = [torch.zeros_like(n) for _ in range(world)]
        dist.all_gather(sizes, n)
        maxn = int(torch.stack(sizes).max().item())
        if maxn == 0:
            return
        pad = torch.zeros((maxn, self.dim), device=self.device, dtype=self.dtype)
        pad[: self.buf.shape[0]] = self.buf
        gathered = [torch.zeros_like(pad) for _ in range(world)]
        dist.all_gather(gathered, pad)
        merged = torch.cat([g[: int(s.item())] for g, s in zip(gathered, sizes)], dim=0)
        self.buf = merged[-self.cap:]


class UDSSelector:
    def __init__(self, vocab: int, cfg: UDSConfig):
        self.cfg = cfg
        self.vocab = vocab
        g = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
        scale = 1.0 / math.sqrt(cfg.svd_proj_dim)
        self.R = (torch.randn(vocab, cfg.svd_proj_dim, generator=g) * scale).to(
            cfg.device, cfg.dtype)
        self.fastjl = FastJL(vocab, cfg)
        self.buffer = DiversityBuffer(cfg)

    @torch.no_grad()
    def score(self, logits: torch.Tensor, attention_mask: torch.Tensor):
        """logits [P,T,V], attention_mask [P,T] -> (s_total[P], z[P,fp2],
        s_intra[P], s_inter[P])."""
        cfg = self.cfg
        P = logits.shape[0]
        logits = logits.to(cfg.dtype)

        # project vocab dim down for a tractable SVD: [P,T,V] @ [V,d] = [P,T,d]
        proj = logits @ self.R
        # mask padding rows so they don't contribute singular mass / summary
        m = attention_mask.unsqueeze(-1).to(cfg.dtype)
        proj = proj * m

        # utility: nuclear norm per sample (sum of singular values of [T,d])
        svals = torch.linalg.svdvals(proj)          # [P, min(T,d)]
        s_intra = svals.sum(dim=1)                   # [P]

        # fingerprint: masked-mean over time in vocab space, then FastJL
        denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(cfg.dtype)
        summary = (logits * m).sum(dim=1) / denom    # [P, V]
        z = self.fastjl(summary)                     # [P, fp2]

        # diversity vs history
        s_inter = self.buffer.mean_l2(z)             # [P]

        # normalize scales so alpha is meaningful across pools
        s_intra_n = _znorm(s_intra)
        s_inter_n = _znorm(s_inter)
        s_total = s_intra_n + cfg.alpha * s_inter_n
        return s_total, z, s_intra, s_inter

    @torch.no_grad()
    def select(self, s_total: torch.Tensor):
        """Global TopK across DDP ranks. Returns local indices to train."""
        cfg = self.cfg
        P = s_total.shape[0]
        if not (dist.is_available() and dist.is_initialized()):
            k = min(cfg.K, P)
            return torch.topk(s_total, k).indices

        # gather all scores, find global Kth threshold, keep local ones above it
        world = dist.get_world_size()
        all_scores = [torch.zeros_like(s_total) for _ in range(world)]
        dist.all_gather(all_scores, s_total)
        flat = torch.cat(all_scores)                 # [world*P]
        kglobal = min(cfg.K * world, flat.numel())
        thresh = torch.topk(flat, kglobal).values.min()
        idx = torch.nonzero(s_total >= thresh, as_tuple=False).squeeze(-1)
        # guard against ties overselecting
        if idx.numel() > cfg.K:
            local_top = torch.topk(s_total[idx], cfg.K).indices
            idx = idx[local_top]
        return idx

    @torch.no_grad()
    def commit(self, z_selected: torch.Tensor) -> None:
        self.buffer.add(z_selected)
        self.buffer.sync_ddp()


def _znorm(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean()
    sd = x.std().clamp(min=1e-6)
    return (x - mu) / sd
