import torch
from src.masks import make_causal_mask, make_pad_mask, make_decoder_self_mask, make_cross_mask

def test_causal_mask_is_lower_triangular():
    x = torch.ones(2, 5, dtype=torch.long)
    m = make_causal_mask(x)  # (1,1,5,5)
    assert m.shape == (1, 1, 5, 5)

    # check upper triangle is False, diagonal/lower is True
    mat = m[0, 0]
    for i in range(5):
        for j in range(5):
            if j > i:
                assert mat[i, j].item() is False
            else:
                assert mat[i, j].item() is True

def test_pad_mask_blocks_pad_tokens():
    pad_idx = 1
    q = torch.tensor([[2, 5, 1, 1]], dtype=torch.long)  # (1,4)
    k = torch.tensor([[7, 1, 9]], dtype=torch.long)     # (1,3)
    m = make_pad_mask(q, k, pad_idx=pad_idx)            # (1,1,4,3)

    # positions where k is pad_idx should be False (masked)
    # k: [7, PAD, 9] -> second column is masked
    assert m.shape == (1, 1, 4, 3)
    assert m[0, 0, 0, 0].item() is True
    assert m[0, 0, 0, 1].item() is False
    assert m[0, 0, 0, 2].item() is True

def test_decoder_self_mask_is_pad_and_causal():
    pad_idx = 1
    tgt = torch.tensor([[2, 4, 5, 1, 1]], dtype=torch.long)  # (1,5)
    m = make_decoder_self_mask(tgt, pad_idx=pad_idx)          # (1,1,5,5)

    # time step 0 cannot attend to future (j>0)
    assert m[0, 0, 0, 1].item() is False

    # pad positions in keys should be False for all queries
    # key positions 3,4 are PAD
    for tq in range(5):
        assert m[0, 0, tq, 3].item() is False
        assert m[0, 0, tq, 4].item() is False

def test_cross_mask_blocks_src_pad():
    pad_idx = 1
    tgt = torch.tensor([[2, 4, 5]], dtype=torch.long)         # (1,3)
    src = torch.tensor([[2, 7, 1, 1]], dtype=torch.long)      # (1,4)
    m = make_cross_mask(tgt, src, pad_idx=pad_idx)            # (1,1,3,4)
    assert m.shape == (1, 1, 3, 4)
    # src key positions 2,3 are PAD -> masked
    assert m[0, 0, 0, 2].item() is False
    assert m[0, 0, 0, 3].item() is False
