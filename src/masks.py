import torch

def make_pad_mask(q: torch.Tensor, k: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    q: (B, Tq) token ids
    k: (B, Tk) token ids
    return: (B, 1, Tq, Tk) bool mask, True means "allowed", False means "masked"
    """
    q_len = q.size(1)
    k_mask = (k != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,Tk)
    return k_mask.expand(-1, 1, q_len, -1)             # (B,1,Tq,Tk)

def make_causal_mask(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T) token ids
    return: (1, 1, T, T) bool mask, True means "allowed" (lower triangular)
    """
    t = x.size(1)
    mask = torch.tril(torch.ones((t, t), device=x.device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)

def make_decoder_self_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    tgt: (B, T)
    return: (B, 1, T, T) bool mask = pad_mask & causal_mask
    """
    pad = make_pad_mask(tgt, tgt, pad_idx)    # (B,1,T,T)
    causal = make_causal_mask(tgt)            # (1,1,T,T)
    return pad & causal

def make_cross_mask(tgt: torch.Tensor, src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    decoder queries (tgt) attend to encoder keys (src)
    return: (B,1,T,S) bool mask
    """
    return make_pad_mask(tgt, src, pad_idx)
