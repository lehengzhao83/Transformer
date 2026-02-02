import torch
import torch.nn as nn
import math

from .masks import make_decoder_self_mask, make_cross_mask, make_pad_mask


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, pad_idx: int = 1, use_amp: bool = True):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            logits = model(src, tgt_in)  # (B,T-1,V)
            V = logits.size(-1)
            loss = criterion(logits.reshape(-1, V), tgt_out.reshape(-1))

        non_pad = (tgt_out != pad_idx).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    avg = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg, 20))
    return avg, ppl


@torch.no_grad()
def greedy_decode(model, src, max_len: int, bos_idx: int = 2, eos_idx: int = 3, pad_idx: int = 1):
    """
    Very basic greedy decoding for demonstration.
    src: (B,S)
    returns decoded token ids including BOS ... EOS (or until max_len)
    """
    device = src.device
    model.eval()

    # encoder
    s_mask = make_pad_mask(src, src, pad_idx)
    enc_out = model.encoder(src, s_mask)

    B = src.size(0)
    decoded = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)  # start with BOS

    for _ in range(max_len - 1):
        t_mask = make_decoder_self_mask(decoded, pad_idx)
        cross_mask = make_cross_mask(decoded, src, pad_idx)

        logits = model.decoder(decoded, enc_out, t_mask, cross_mask)  # (B,T,V)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)    # (B,1)
        decoded = torch.cat([decoded, next_token], dim=1)

        if (next_token == eos_idx).all():
            break

    return decoded
