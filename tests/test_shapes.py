import torch
from src.model import Transformer, MultiHeadAttention, TransformerEmbedding

def test_embedding_shape():
    device = torch.device("cpu")
    B, T = 2, 7
    vocab = 50
    d_model = 32
    emb = TransformerEmbedding(vocab, d_model, max_len=64, drop_prob=0.0, device=device, padding_idx=1)
    x = torch.randint(0, vocab, (B, T), dtype=torch.long)
    y = emb(x)
    assert y.shape == (B, T, d_model)

def test_mha_shapes_self_attention():
    B, T, D = 2, 5, 32
    H = 4
    mha = MultiHeadAttention(D, H, attn_dropout=0.0)
    x = torch.randn(B, T, D)
    y = mha(x, x, x, mask=None)
    assert y.shape == (B, T, D)

def test_mha_shapes_cross_attention():
    B, Tq, Tk, D = 2, 4, 7, 32
    H = 4
    mha = MultiHeadAttention(D, H, attn_dropout=0.0)
    q = torch.randn(B, Tq, D)
    k = torch.randn(B, Tk, D)
    v = torch.randn(B, Tk, D)
    y = mha(q, k, v, mask=None)
    assert y.shape == (B, Tq, D)

def test_transformer_output_shape():
    device = torch.device("cpu")
    B, S, T = 2, 9, 8
    vocab = 100

    model = Transformer(
        src_vocab_size=vocab,
        tgt_vocab_size=vocab,
        max_len=64,
        d_model=32,
        ffn_hidden=64,
        n_head=4,
        n_layer=2,
        device=device,
        dropout=0.0,
        pad_idx=1,
    )

    src = torch.randint(0, vocab, (B, S), dtype=torch.long)
    tgt_in = torch.randint(0, vocab, (B, T), dtype=torch.long)
    logits = model(src, tgt_in)
    assert logits.shape == (B, T, vocab)
