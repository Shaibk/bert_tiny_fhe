import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_additive_mask(attention_mask: torch.Tensor, dtype: torch.dtype):
    """
    attention_mask: [B,T] 1=valid, 0=pad
    return: [B,1,1,T] additive mask (0 or -1e4)
    """
    neg = torch.tensor(-1e4, device=attention_mask.device, dtype=dtype)
    return (1.0 - attention_mask.to(dtype))[:, None, None, :] * neg

class AttnSoftmax(nn.Module):
    def __init__(self, head_dim: int, dropout=0.0):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, attention_mask=None):
        # q,k,v: [B,H,T,D]
        scores = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,T,T]
        if attention_mask is not None:
            scores = scores + make_additive_mask(attention_mask, scores.dtype)
        p = F.softmax(scores, dim=-1)
        p = self.drop(p)
        out = p @ v
        return out, p

class Attn2Quad(nn.Module):
    """
    2Quad: (x+c)^2 / sum (x+c)^2
    """
    def __init__(self, head_dim: int, c=3.0, eps=1e-6, dropout=0.0):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_dim)
        self.c = float(c)
        self.eps = float(eps)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, attention_mask=None):
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores + make_additive_mask(attention_mask, scores.dtype)

        x = scores + self.c
        num = x * x               # 一次平方（方案一）
        den = num.sum(dim=-1, keepdim=True) + self.eps
        p = num / den
        p = self.drop(p)
        out = p @ v
        return out, p

class AttnPowerP2(nn.Module):
    """
    PowerSoftmax p=2: x^2 / sum x^2
    """
    def __init__(self, head_dim: int, eps=1e-6, dropout=0.0):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_dim)
        self.eps = float(eps)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, attention_mask=None):
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores + make_additive_mask(attention_mask, scores.dtype)

        num = scores * scores      # 一次平方（方案一）
        den = num.sum(dim=-1, keepdim=True) + self.eps
        p = num / den
        p = self.drop(p)
        out = p @ v
        return out, p
