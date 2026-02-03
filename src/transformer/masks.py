from __future__ import annotations

import torch
from torch import Tensor


def make_pad_mask(q: Tensor, k: Tensor, pad_idx: int = 1) -> Tensor:
    """
    Create padding mask for attention.

    Args:
      q: (B, Tq) token ids
      k: (B, Tk) token ids

    Returns:
      mask: (B, 1, Tq, Tk) bool tensor
            True means "allowed", False means "masked".
    """
    if q.dim() != 2 or k.dim() != 2:
        msg = f"q and k must be rank-2 tensors (B,T). Got q={tuple(q.shape)} k={tuple(k.shape)}"
        raise ValueError(msg)
    if q.size(0) != k.size(0):
        msg = f"Batch size mismatch: q has B={int(q.size(0))}, k has B={int(k.size(0))}"
        raise ValueError(msg)

    tq = int(q.size(1))
    # (B, 1, 1, Tk)
    k_keep = (k != pad_idx).to(dtype=torch.bool).unsqueeze(1).unsqueeze(2)
    # (B, 1, Tq, Tk)
    return k_keep.expand(-1, 1, tq, -1)


def make_causal_mask(x: Tensor) -> Tensor:
    """
    Create causal (lower-triangular) mask.

    Args:
      x: (B, T) token ids (only length and device are used)

    Returns:
      mask: (1, 1, T, T) bool tensor, True means "allowed".
    """
    if x.dim() != 2:
        msg = f"x must be rank-2 tensor (B,T). Got x={tuple(x.shape)}"
        raise ValueError(msg)

    t = int(x.size(1))
    if t <= 0:
        raise ValueError("sequence length must be positive")

    mask = torch.ones((t, t), device=x.device, dtype=torch.bool).tril()
    return mask.unsqueeze(0).unsqueeze(0)


def make_decoder_self_mask(tgt: Tensor, pad_idx: int = 1) -> Tensor:
    """
    Decoder self-attention mask = padding mask & causal mask.

    Args:
      tgt: (B, T) token ids

    Returns:
      mask: (B, 1, T, T) bool tensor, True means "allowed".
    """
    pad = make_pad_mask(tgt, tgt, pad_idx)  # (B,1,T,T)
    causal = make_causal_mask(tgt)          # (1,1,T,T)
    return pad & causal


def make_cross_mask(tgt: Tensor, src: Tensor, pad_idx: int = 1) -> Tensor:
    """
    Cross-attention mask for decoder queries (tgt) attending to encoder keys (src).

    Args:
      tgt: (B, T) token ids
      src: (B, S) token ids

    Returns:
      mask: (B, 1, T, S) bool tensor, True means "allowed".
    """
    return make_pad_mask(tgt, src, pad_idx)
