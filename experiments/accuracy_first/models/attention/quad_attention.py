from __future__ import annotations
import math
import torch
import torch.nn as nn

class QuadAttention(nn.Module):
    """
    2Quad 近似 Softmax:
      P_i = (x_i + c)^2 / sum_j (x_j + c)^2

    注意：这里实现的是“明文/浮点训练可用版本”。
    后续回到FHE时，分子平方是一层乘法；分母倒数通常放交互/MPC/预处理，或用低阶近似。
    """
    def __init__(self, head_dim: int, c: float = 3.0, eps: float = 1e-6, dropout: float = 0.0):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_dim)
        self.c = float(c)
        self.eps = float(eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,T,T]
        if attn_mask is not None:
            scores = scores + attn_mask

        x = scores + self.c
        num = x * x  # 核心：一层平方
        den = num.sum(dim=-1, keepdim=True) + self.eps
        prob = num / den
        prob = self.dropout(prob)
        out = torch.matmul(prob, v)
        return out, prob
