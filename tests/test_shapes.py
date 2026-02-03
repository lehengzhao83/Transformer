from __future__ import annotations

import torch
from torch import Tensor

from transformer.model import MultiHeadAttention, Transformer, TransformerEmbedding


def test_embedding_shape() -> None:
    bsz, seq_len = 2, 7
    vocab = 50
    d_model = 32

    emb = TransformerEmbedding(vocab, d_model, max_len=64, drop_prob=0.0, padding_idx=1)
    x: Tensor = torch.randint(0, vocab, (bsz, seq_len), dtype=torch.long)
    y = emb(x)
    assert y.shape == (bsz, seq_len, d_model)


def test_mha_shapes_self_attention() -> None:
    bsz, seq_len, d_model = 2, 5, 32
    n_head = 4

    mha = MultiHeadAttention(d_model, n_head, attn_dropout=0.0)
    x: Tensor = torch.randn(bsz, seq_len, d_model)
    y = mha(x, x, x, mask=None)
    assert y.shape == (bsz, seq_len, d_model)


def test_mha_shapes_cross_attention() -> None:
    bsz, tq, tk, d_model = 2, 4, 7, 32
    n_head = 4

    mha = MultiHeadAttention(d_model, n_head, attn_dropout=0.0)
    q: Tensor = torch.randn(bsz, tq, d_model)
    k: Tensor = torch.randn(bsz, tk, d_model)
    v: Tensor = torch.randn(bsz, tk, d_model)
    y = mha(q, k, v, mask=None)
    assert y.shape == (bsz, tq, d_model)


def test_transformer_output_shape() -> None:
    device = torch.device("cpu")
    bsz, s_len, t_len = 2, 9, 8
    vocab = 100

    model = Transformer(
        src_vocab_size=vocab,
        tgt_vocab_size=vocab,
        max_len=64,
        d_model=32,
        ffn_hidden=64,
        n_head=4,
        n_layer=2,
        device=device,  # kept for backward compatibility; model doesn't require it now
        dropout=0.0,
        pad_idx=1,
    )

    src: Tensor = torch.randint(0, vocab, (bsz, s_len), dtype=torch.long)
    tgt_in: Tensor = torch.randint(0, vocab, (bsz, t_len), dtype=torch.long)
    logits = model(src, tgt_in)
    assert logits.shape == (bsz, t_len, vocab)
