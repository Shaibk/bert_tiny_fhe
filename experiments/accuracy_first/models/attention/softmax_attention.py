from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxAttention(nn.Module):
    def __init__(self, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None):
        # q,k,v: [B, H, T, Dh]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,T,T]
        if attn_mask is not None:
            scores = scores + attn_mask
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        out = torch.matmul(prob, v)  # [B,H,T,Dh]
        return out, prob
