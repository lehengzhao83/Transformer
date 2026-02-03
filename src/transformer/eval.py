from __future__ import annotations

import math
from typing import Iterable, Protocol, Tuple

import torch
from torch import Tensor

from .masks import make_cross_mask, make_decoder_self_mask, make_pad_mask


class _ModelForEval(Protocol):
    def eval(self) -> None: ...
    def __call__(self, src: Tensor, tgt_in: Tensor) -> Tensor: ...


class _ModelForDecode(Protocol):
    def eval(self) -> None: ...
    # these match your current usage: model.encoder / model.decoder
    def encoder(self, src: Tensor, src_mask: Tensor) -> Tensor: ...
    def decoder(self, tgt: Tensor, enc_out: Tensor, tgt_mask: Tensor, cross_mask: Tensor) -> Tensor: ...


class _Criterion(Protocol):
    def __call__(self, input: Tensor, target: Tensor) -> Tensor: ...


def _as_device(device: torch.device | str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _autocast_enabled(device: torch.device, use_amp: bool) -> bool:
    return bool(use_amp and device.type == "cuda")


@torch.no_grad()
def evaluate_loss(
    model: _ModelForEval,
    loader: Iterable[tuple[Tensor, Tensor]],
    criterion: _Criterion,
    device: torch.device | str,
    *,
    pad_idx: int = 1,
    use_amp: bool = True,
) -> tuple[float, float]:
    """
    Compute token-average loss (ignoring PAD) and perplexity.

    loader yields:
      src: (B, S) LongTensor
      tgt: (B, T) LongTensor
    """
    dev = _as_device(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(dev)
        tgt = tgt.to(dev)

        if tgt.dim() != 2:
            msg = f"tgt must be rank-2 (B,T); got shape={tuple(tgt.shape)}"
            raise ValueError(msg)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        # AMP: prefer torch.amp.autocast on newer PyTorch, fallback to cuda.amp.autocast
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=_autocast_enabled(dev, use_amp))
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=_autocast_enabled(dev, use_amp))

        with autocast_ctx:
            logits = model(src, tgt_in)  # (B, T-1, V)
            if logits.dim() != 3:
                msg = f"logits must be rank-3 (B,T,V); got shape={tuple(logits.shape)}"
                raise ValueError(msg)
            v = int(logits.size(-1))
            loss = criterion(logits.reshape(-1, v), tgt_out.reshape(-1))

        non_pad = int((tgt_out != pad_idx).sum().item())
        total_loss += float(loss.detach().cpu().item()) * non_pad
        total_tokens += non_pad

    avg = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg, 20.0))  # cap to avoid inf
    return float(avg), float(ppl)


@torch.no_grad()
def greedy_decode(
    model: _ModelForDecode,
    src: Tensor,
    *,
    max_len: int,
    bos_idx: int = 2,
    eos_idx: int = 3,
    pad_idx: int = 1,
) -> Tensor:
    """
    Very basic greedy decoding for demonstration.

    Args:
      src: (B, S) LongTensor
      returns: (B, <=max_len) LongTensor including BOS ... EOS (or until max_len)

    Note:
      This uses your model.encoder / model.decoder API and your masks helpers.
    """
    if max_len <= 1:
        raise ValueError("max_len must be >= 2")

    device = src.device
    model.eval()

    # encoder
    src_mask = make_pad_mask(src, src, pad_idx)
    enc_out = model.encoder(src, src_mask)

    bsz = int(src.size(0))
    decoded = torch.full((bsz, 1), bos_idx, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_decoder_self_mask(decoded, pad_idx)
        cross_mask = make_cross_mask(decoded, src, pad_idx)

        logits = model.decoder(decoded, enc_out, tgt_mask, cross_mask)  # (B, T, V)
        if logits.dim() != 3:
            msg = f"decoder logits must be rank-3 (B,T,V); got shape={tuple(logits.shape)}"
            raise ValueError(msg)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (B, 1)
        decoded = torch.cat([decoded, next_token], dim=1)

        if bool((next_token == eos_idx).all().item()):
            break

    return decoded
