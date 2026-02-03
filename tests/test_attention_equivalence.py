from __future__ import annotations

import torch
from torch import Tensor, nn

from transformer.model import MultiHeadAttention


def copy_weights_ours_to_torch(ours: MultiHeadAttention, torch_mha: nn.MultiheadAttention) -> None:
    """
    Align parameters:
      ours.w_q, ours.w_k, ours.w_v, ours.out_proj
    -> torch_mha.in_proj_weight/bias and out_proj.weight/bias
    """
    d_model = ours.d_model
    with torch.no_grad():
        # in_proj_weight: (3*d_model, d_model)
        torch_mha.in_proj_weight[:d_model, :] = ours.w_q.weight
        torch_mha.in_proj_weight[d_model : 2 * d_model, :] = ours.w_k.weight
        torch_mha.in_proj_weight[2 * d_model : 3 * d_model, :] = ours.w_v.weight

        torch_mha.in_proj_bias[:d_model] = ours.w_q.bias
        torch_mha.in_proj_bias[d_model : 2 * d_model] = ours.w_k.bias
        torch_mha.in_proj_bias[2 * d_model : 3 * d_model] = ours.w_v.bias

        torch_mha.out_proj.weight[:] = ours.out_proj.weight
        torch_mha.out_proj.bias[:] = ours.out_proj.bias


def test_attention_equivalence_no_mask() -> None:
    torch.manual_seed(0)

    bsz, seq_len, d_model, n_head = 2, 5, 32, 4
    x: Tensor = torch.randn(bsz, seq_len, d_model)

    ours = MultiHeadAttention(d_model, n_head, attn_dropout=0.0)
    ours.eval()

    torch_mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=0.0, batch_first=True)
    torch_mha.eval()

    copy_weights_ours_to_torch(ours, torch_mha)

    y_ours = ours(x, x, x, mask=None)
    y_torch, _ = torch_mha(x, x, x, need_weights=False)

    max_abs_err = float((y_ours - y_torch).abs().max().item())
    assert max_abs_err < 1e-5
