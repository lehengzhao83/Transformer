from __future__ import annotations

import torch
from torch import Tensor


def make_pad_mask(q: Tensor, k: Tensor, pad_idx: int = 1) -> Tensor:
    """
    q: (B, Tq) token ids
    k: (B, Tk) token ids

    return: (B, 1, Tq, Tk) bool mask
      True means "allowed", False means "masked".
    """
    if q.dim() != 2 or k.dim() != 2:
        msg = f"q and k must be rank-2 (B,T). Got q={tuple(q.shape)} k={tuple(k.shape)}"
        raise ValueError(msg)
    if q.size(0) != k.size(0):
        msg = f"Batch mismatch: q B={int(q.size(0))}, k B={int(k.size(0))}"
        raise ValueError(msg)

    tq = int(q.size(1))
    k_keep = (k != pad_idx).to(dtype=torch.bool).unsqueeze(1).unsqueeze(2)  # (B,1,1,Tk)
    return k_keep.expand(-1, 1, tq, -1)  # (B,1,Tq,Tk)


def make_causal_mask(x: Tensor) -> Tensor:
    """
    x: (B, T) token ids (only T and device are used)

    return: (1, 1, T, T) bool mask (lower triangular), True means "allowed".
    """
    if x.dim() != 2:
        msg = f"x must be rank-2 (B,T). Got x={tuple(x.shape)}"
        raise ValueError(msg)

    t = int(x.size(1))
    if t <= 0:
        raise ValueError("sequence length must be positive")

    mask = torch.ones((t, t), device=x.device, dtype=torch.bool).tril()
    return mask.unsqueeze(0).unsqueeze(0)


def make_decoder_self_mask(tgt: Tensor, pad_idx: int = 1) -> Tensor:
    """
    tgt: (B, T)
    return: (B, 1, T, T) bool mask = pad_mask & causal_mask
    """
    pad = make_pad_mask(tgt, tgt, pad_idx)
    causal = make_causal_mask(tgt)
    return pad & causal


def make_cross_mask(tgt: Tensor, src: Tensor, pad_idx: int = 1) -> Tensor:
    """
    decoder queries (tgt) attend to encoder keys (src)
    return: (B, 1, T, S) bool mask
    """
    return make_pad_mask(tgt, src, pad_idx)
