from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import Tensor, nn

from .masks import make_cross_mask, make_decoder_self_mask, make_pad_mask
from .model import Transformer


def _autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()


@torch.no_grad()
def evaluate_loss(
    model: Transformer,
    loader: torch.utils.data.DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    *,
    pad_idx: int = 1,
    use_amp: bool = True,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        with _autocast_context(device, use_amp):
            logits = model(src, tgt_in)  # (B,T-1,V)
            v = int(logits.size(-1))
            loss = criterion(logits.reshape(-1, v), tgt_out.reshape(-1))

        non_pad = int((tgt_out != pad_idx).sum().item())
        total_loss += float(loss.item()) * non_pad
        total_tokens += non_pad

    avg = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg, 20.0))
    return float(avg), float(ppl)


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: Tensor,
    *,
    max_len: int,
    bos_idx: int = 2,
    eos_idx: int = 3,
    pad_idx: int = 1,
) -> Tensor:
    """
    Very basic greedy decoding for demonstration.
    src: (B,S)
    returns decoded token ids including BOS ... EOS (or until max_len)
    """
    device = src.device
    model.eval()

    s_mask = make_pad_mask(src, src, pad_idx)
    enc_out = model.encoder(src, s_mask)

    bsz = int(src.size(0))
    decoded = torch.full((bsz, 1), bos_idx, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        t_mask = make_decoder_self_mask(decoded, pad_idx)
        cross_mask = make_cross_mask(decoded, src, pad_idx)

        logits = model.decoder(decoded, enc_out, t_mask, cross_mask)  # (B,T,V)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B,1)
        decoded = torch.cat([decoded, next_token], dim=1)

        if bool((next_token == eos_idx).all().item()):
            break

    return decoded
