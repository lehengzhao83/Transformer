import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import make_pad_mask, make_decoder_self_mask, make_cross_mask


# -------------------------
# Embeddings
# -------------------------

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 1):
        super().__init__(vocab_size, d_model, padding_idx=padding_idx)


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding (non-trainable)."""
    def __init__(self, d_model: int, max_len: int, device: torch.device):
        super().__init__()
        encoding = torch.zeros(max_len, d_model, device=device)
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)  # (max_len,1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()       # (d_model/2,)

        div = 10000 ** (_2i / d_model)
        encoding[:, 0::2] = torch.sin(pos / div)
        encoding[:, 1::2] = torch.cos(pos / div)

        self.register_buffer("encoding", encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        seq_len = x.size(1)
        return self.encoding[:seq_len, :].unsqueeze(0)  # (1,T,D)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int,
                 drop_prob: float, device: torch.device, padding_idx: int = 1):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.tok_emb(x)      # (B,T,D)
        pos = self.pos_emb(x)      # (1,T,D)
        return self.dropout(tok + pos)


# -------------------------
# Core Blocks
# -------------------------

class LayerNorm(nn.Module):
    """Custom LayerNorm (matching your style)."""
    def __init__(self, d_model: int, eps: float = 1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * out + self.beta


class MultiHeadAttention(nn.Module):
    """
    mask: bool, broadcastable to (B, H, Tq, Tk)
    True allowed, False masked
    """
    def __init__(self, d_model: int, n_head: int, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        b, tq, d = q.shape
        _, tk, _ = k.shape
        _, tv, _ = v.shape
        assert tk == tv, "k and v length must match"

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = q.view(b, tq, self.n_head, self.d_k).permute(0, 2, 1, 3)  # (B,H,Tq,Dk)
        k = k.view(b, tk, self.n_head, self.d_k).permute(0, 2, 1, 3)  # (B,H,Tk,Dk)
        v = v.view(b, tv, self.n_head, self.d_k).permute(0, 2, 1, 3)  # (B,H,Tk,Dk)

        scores = (q @ k.transpose(2, 3)) / math.sqrt(self.d_k)         # (B,H,Tq,Tk)

        if mask is not None:
            # mask: True allowed, False masked
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v                                                 # (B,H,Tq,Dk)

        out = out.permute(0, 2, 1, 3).contiguous().view(b, tq, d)       # (B,Tq,D)
        return self.out_proj(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


# -------------------------
# Encoder / Decoder Layers
# -------------------------

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head, attn_dropout=dropout)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, s_mask: torch.Tensor = None) -> torch.Tensor:
        _x = x
        x = self.attn(x, x, x, s_mask)
        x = self.drop1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.drop2(x)
        x = self.norm2(x + _x)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, d_model: int, ffn_hidden: int,
                 n_head: int, n_layer: int, device: torch.device, dropout: float = 0.1,
                 pad_idx: int = 1):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout, device, pad_idx)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, ffn_hidden, n_head, dropout)
            for _ in range(n_layer)
        ])

    def forward(self, src: torch.Tensor, s_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(src)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, dropout: float = 0.1):
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

    def forward(self, dec: torch.Tensor, enc: torch.Tensor,
                t_mask: torch.Tensor = None, s_mask: torch.Tensor = None) -> torch.Tensor:
        _x = dec
        x = self.self_attn(dec, dec, dec, t_mask)
        x = self.drop1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.cross_attn(x, enc, enc, s_mask)
        x = self.drop2(x)
        x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.drop3(x)
        x = self.norm3(x + _x)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, d_model: int, ffn_hidden: int,
                 n_head: int, n_layer: int, device: torch.device, dropout: float = 0.1,
                 pad_idx: int = 1):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout, device, pad_idx)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, ffn_hidden, n_head, dropout)
            for _ in range(n_layer)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_in: torch.Tensor, enc_out: torch.Tensor,
                t_mask: torch.Tensor = None, s_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(tgt_in)
        for layer in self.layers:
            x = layer(x, enc_out, t_mask, s_mask)
        return self.fc(x)  # (B,T,V)


# -------------------------
# Full Transformer
# -------------------------

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, max_len: int,
                 d_model: int, ffn_hidden: int, n_head: int, n_layer: int,
                 device: torch.device, dropout: float = 0.1, pad_idx: int = 1):
        super().__init__()
        self.pad_idx = pad_idx
        self.encoder = Encoder(src_vocab_size, max_len, d_model, ffn_hidden, n_head, n_layer, device, dropout, pad_idx)
        self.decoder = Decoder(tgt_vocab_size, max_len, d_model, ffn_hidden, n_head, n_layer, device, dropout, pad_idx)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """
        src: (B,S) long
        tgt_in: (B,T) long  (teacher forcing input, usually includes BOS)
        return: logits (B,T,V)
        """
        s_mask = make_pad_mask(src, src, self.pad_idx)             # (B,1,S,S)
        t_mask = make_decoder_self_mask(tgt_in, self.pad_idx)      # (B,1,T,T)
        cross_mask = make_cross_mask(tgt_in, src, self.pad_idx)    # (B,1,T,S)

        enc_out = self.encoder(src, s_mask)
        logits = self.decoder(tgt_in, enc_out, t_mask, cross_mask)
        return logits
