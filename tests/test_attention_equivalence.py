import torch
import torch.nn as nn
from src.model import MultiHeadAttention

def copy_weights_ours_to_torch(ours: MultiHeadAttention, torch_mha: nn.MultiheadAttention):
    """
    Align parameters:
    ours.w_q, ours.w_k, ours.w_v, ours.out_proj
    -> torch_mha.in_proj_weight/bias and out_proj.weight/bias
    """
    d = ours.d_model
    with torch.no_grad():
        # in_proj_weight: (3d, d)
        torch_mha.in_proj_weight[:d, :] = ours.w_q.weight
        torch_mha.in_proj_weight[d:2*d, :] = ours.w_k.weight
        torch_mha.in_proj_weight[2*d:3*d, :] = ours.w_v.weight

        torch_mha.in_proj_bias[:d] = ours.w_q.bias
        torch_mha.in_proj_bias[d:2*d] = ours.w_k.bias
        torch_mha.in_proj_bias[2*d:3*d] = ours.w_v.bias

        torch_mha.out_proj.weight[:] = ours.out_proj.weight
        torch_mha.out_proj.bias[:] = ours.out_proj.bias

def test_attention_equivalence_no_mask():
    torch.manual_seed(0)

    B, T, D, H = 2, 5, 32, 4
    x = torch.randn(B, T, D)

    ours = MultiHeadAttention(D, H, attn_dropout=0.0)
    ours.eval()

    torch_mha = nn.MultiheadAttention(embed_dim=D, num_heads=H, dropout=0.0, batch_first=True)
    torch_mha.eval()

    copy_weights_ours_to_torch(ours, torch_mha)

    y_ours = ours(x, x, x, mask=None)
    y_torch, _ = torch_mha(x, x, x, need_weights=False)

    max_abs_err = (y_ours - y_torch).abs().max().item()
    assert max_abs_err < 1e-5
