from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .masks import make_cross_mask, make_decoder_self_mask, make_pad_mask


# -------------------------
# Config (optional but helps typing & future scaling)
# -------------------------


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    max_len: int
    d_model: int
    ffn_hidden: int
    n_head: int
    n_layer: int
    dropout: float = 0.1
    pad_idx: int = 1


# -------------------------
# Embeddings
# -------------------------


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 1) -> None:
        super().__init__(vocab_size, d_model, padding_idx=padding_idx)


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding (non-trainable)."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if max_len <= 0:
            raise ValueError("max_len must be positive")

        # Build on CPU; buffer will move with module.to(device)
        encoding = torch.zeros((max_len, d_model), dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        idx = torch.arange(0, d_model, step=2, dtype=torch.float32)  # (d_model/2,)

        div = torch.pow(torch.tensor(10000.0, dtype=torch.float32), idx / float(d_model))
        encoding[:, 0::2] = torch.sin(pos / div)
        encoding[:, 1::2] = torch.cos(pos / div)

        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T) token ids
        if x.dim() != 2:
            msg = f"x must be rank-2 (B,T) token ids, got shape={tuple(x.shape)}"
            raise ValueError(msg)
        seq_len = int(x.size(1))
        return self.encoding[:seq_len, :].unsqueeze(0)  # (1, T, D)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, drop_prob: float, padding_idx: int = 1) -> None:
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.pos_emb = PositionalEmbedding(d_model, max_len)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T) long
        tok = self.tok_emb(x)  # (B, T, D)
        pos = self.pos_emb(x)  # (1, T, D)
        return self.dropout(tok + pos)


# -------------------------
# Core Blocks
# -------------------------


class LayerNorm(nn.Module):
    """Custom LayerNorm (matching your style)."""

    def __init__(self, d_model: int, eps: float = 1e-12) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * out + self.beta


class MultiHeadAttention(nn.Module):
    """
    mask: bool tensor broadcastable to (B, H, Tq, Tk)
    True = allowed, False = masked
    """

    def __init__(self, d_model: int, n_head: int, attn_dropout: float = 0.0) -> None:
        super().__init__()
        if n_head <= 0:
            raise ValueError("n_head must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError("q/k/v must be rank-3 tensors (B,T,D)")

        b, tq, d = q.shape
        _, tk, dk = k.shape
        _, tv, dv = v.shape

        if d != self.d_model or dk != self.d_model or dv != self.d_model:
            raise ValueError("Last dim of q/k/v must equal d_model")
        if tk != tv:
            raise ValueError("k and v length must match")

        q_proj = self.w_q(q)
        k_proj = self.w_k(k)
        v_proj = self.w_v(v)

        # (B, H, T, Dk)
        qh = q_proj.view(b, tq, self.n_head, self.d_k).permute(0, 2, 1, 3)
        kh = k_proj.view(b, tk, self.n_head, self.d_k).permute(0, 2, 1, 3)
        vh = v_proj.view(b, tv, self.n_head, self.d_k).permute(0, 2, 1, 3)

        # (B, H, Tq, Tk)
        scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dtype is not torch.bool:
                raise TypeError("mask must be a bool tensor")
            # Use finfo.min (safe for fp16/bf16) rather than -inf to avoid some kernels issues.
            fill_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, fill_value)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ vh  # (B, H, Tq, Dk)

        out = out.permute(0, 2, 1, 3).contiguous().view(b, tq, d)  # (B, Tq, D)
        return self.out_proj(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model <= 0 or hidden <= 0:
            raise ValueError("d_model and hidden must be positive")

        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


# -------------------------
# Encoder / Decoder Layers
# -------------------------


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head, attn_dropout=dropout)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, s_mask: Tensor | None = None) -> Tensor:
        res = x
        x = self.attn(x, x, x, s_mask)
        x = self.drop1(x)
        x = self.norm1(x + res)

        res = x
        x = self.ffn(x)
        x = self.drop2(x)
        x = self.norm2(x + res)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        ffn_hidden: int,
        n_head: int,
        n_layer: int,
        dropout: float = 0.1,
        pad_idx: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout, padding_idx=pad_idx)
        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden, n_head, dropout) for _ in range(n_layer)])

    def forward(self, src: Tensor, s_mask: Tensor | None = None) -> Tensor:
        x = self.embedding(src)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, attn_dropout=dropout)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.cross_attn = MultiHeadAttention(d_model, n_head, attn_dropout=dropout)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm3 = LayerNorm(d_model)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, dec: Tensor, enc: Tensor, t_mask: Tensor | None = None, s_mask: Tensor | None = None) -> Tensor:
        res = dec
        x = self.self_attn(dec, dec, dec, t_mask)
        x = self.drop1(x)
        x = self.norm1(x + res)

        res = x
        x = self.cross_attn(x, enc, enc, s_mask)
        x = self.drop2(x)
        x = self.norm2(x + res)

        res = x
        x = self.ffn(x)
        x = self.drop3(x)
        x = self.norm3(x + res)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        ffn_hidden: int,
        n_head: int,
        n_layer: int,
        dropout: float = 0.1,
        pad_idx: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout, padding_idx=pad_idx)
        self.layers = nn.ModuleList([DecoderLayer(d_model, ffn_hidden, n_head, dropout) for _ in range(n_layer)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_in: Tensor, enc_out: Tensor, t_mask: Tensor | None = None, s_mask: Tensor | None = None) -> Tensor:
        x = self.embedding(tgt_in)
        for layer in self.layers:
            x = layer(x, enc_out, t_mask, s_mask)
        return self.fc(x)  # (B, T, V)


# -------------------------
# Full Transformer
# -------------------------


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_len: int,
        d_model: int,
        ffn_hidden: int,
        n_head: int,
        n_layer: int,
        device: torch.device | None = None,
        dropout: float = 0.1,
        pad_idx: int = 1,
    ) -> None:
        """
        device is optional: module buffers/params can be moved via model.to(device).
        Kept for backward compatibility with your original signature.
        """
        super().__init__()
        _ = device  # device is not required anymore; kept to avoid breaking caller code
        self.pad_idx = pad_idx

        self.encoder = Encoder(
            src_vocab_size,
            max_len,
            d_model,
            ffn_hidden,
            n_head,
            n_layer,
            dropout=dropout,
            pad_idx=pad_idx,
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            max_len,
            d_model,
            ffn_hidden,
            n_head,
            n_layer,
            dropout=dropout,
            pad_idx=pad_idx,
        )

    @classmethod
    def from_config(cls, cfg: TransformerConfig, device: torch.device | None = None) -> "Transformer":
        return cls(
            cfg.src_vocab_size,
            cfg.tgt_vocab_size,
            cfg.max_len,
            cfg.d_model,
            cfg.ffn_hidden,
            cfg.n_head,
            cfg.n_layer,
            device=device,
            dropout=cfg.dropout,
            pad_idx=cfg.pad_idx,
        )

    def forward(self, src: Tensor, tgt_in: Tensor) -> Tensor:
        """
        Args:
          src: (B, S) long
          tgt_in: (B, T) long  (teacher forcing input, usually includes BOS)

        Returns:
          logits: (B, T, V)
        """
        if src.dim() != 2 or tgt_in.dim() != 2:
            raise ValueError("src and tgt_in must be rank-2 tensors (B,T)")

        s_mask = make_pad_mask(src, src, self.pad_idx)            # (B,1,S,S)
        t_mask = make_decoder_self_mask(tgt_in, self.pad_idx)     # (B,1,T,T)
        cross_mask = make_cross_mask(tgt_in, src, self.pad_idx)   # (B,1,T,S)

        enc_out = self.encoder(src, s_mask)
        logits = self.decoder(tgt_in, enc_out, t_mask, cross_mask)
        return logits
