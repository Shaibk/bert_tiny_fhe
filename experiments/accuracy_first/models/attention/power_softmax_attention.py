from __future__ import annotations
import math
import torch
import torch.nn as nn

class PowerSoftmaxAttention(nn.Module):
    """
    PowerSoftmax（这里用 p=2 以满足你们严格8层预算）:
      P_i = x_i^p / sum_j x_j^p

    注意：
    - p=2: 一次平方（+1）
    - 这里对 scores 做了 shift/clip 的可选位置（默认不做），训练时可在外部加稳定项。
    """
    def __init__(self, head_dim: int, p: int = 2, eps: float = 1e-6, dropout: float = 0.0):
        super().__init__()
        assert p % 2 == 0 and p >= 2, "p must be an even integer >= 2"
        self.scale = 1.0 / math.sqrt(head_dim)
        self.p = int(p)
        self.eps = float(eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores + attn_mask

        x = scores
        # p=2：一次平方；更高p可用 repeated squaring，但方案一固定p=2
        num = x * x
        den = num.sum(dim=-1, keepdim=True) + self.eps
        prob = num / den
        prob = self.dropout(prob)
        out = torch.matmul(prob, v)
        return out, prob
