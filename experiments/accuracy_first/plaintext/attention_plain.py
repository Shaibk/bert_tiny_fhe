import torch
import torch.nn as nn
import math

class SoftmaxAttention(nn.Module):
    """
    标准的 Softmax Attention (Teacher 用)
    Score = Softmax(QK^T / sqrt(d))
    """
    def __init__(self, head_dim, dropout=0.1):
        super().__init__()
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attention_mask=None, tau=None):
        # q, k, v: [B, H, L, D]
        
        # 1. Scaled Dot-Product
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # 2. Masking (Additive Mask for Softmax)
        if attention_mask is not None:
            # attention_mask usually is 1 for valid, 0 for pad
            # We want to add -inf to pad positions
            # mask shape: [B, L] -> [B, 1, 1, L]
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)

        # 3. Softmax
        attn_probs = self.softmax(scores)
        attn_probs = self.dropout(attn_probs)
        
        # 4. Output
        out = torch.matmul(attn_probs, v)
        return out, attn_probs


class Attn2Quad(nn.Module):
    """
    FHE 友好的 2-Quad Attention (Student 用)
    Score = ((QK^T + c)^2 * static_scale) / tau
    """
    def __init__(self, head_dim, dropout=0.1, c=4.0, eps=1e-6):
        super().__init__()
        self.c = c
        self.eps = eps
        self.dropout = nn.Dropout(dropout)
        
        # [关键保留] 静态缩放因子，防止数值爆炸
        # 这个值在导出 FHE 权重时会被融合
        self.static_scale = 0.01 

    def forward(self, q, k, v, attention_mask=None, tau=None):
        # q, k: [B, H, L, D]
        
        # 1. 计算原始分数 QK^T
        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        
        # 2. 应用 2Quad: (x + c)^2 -> 值域约 [16, 100+]
        attn_scores = (attn_scores + self.c).pow(2)
        
        # 3. 静态缩放 (把数值拉回正常范围)
        attn_scores = attn_scores * self.static_scale
        
        # 4. 动态微调 (Learnable Tau)
        if tau is not None:
            # view for broadcasting: [B, H, L, L] / [H] -> [1, H, 1, 1]
            attn_scores = attn_scores / tau.view(1, -1, 1, 1)

        # 5. Masking (Multiplicative Mask for Polynomial)
        if attention_mask is not None:
            # mask: 1 keep, 0 remove. 直接乘即可
            attn_scores = attn_scores * attention_mask.unsqueeze(1).unsqueeze(2)

        # 6. Output
        out = torch.matmul(attn_scores, v)
        out = self.dropout(out)
        # attn_scores: [B,H,L,L]
        print("PT head0 row0 sum:", attn_scores[0,0,0,:].sum().item())
        print("PT head1 row0 sum:", attn_scores[0,1,0,:].sum().item())
        print("PT head0/1 row0 max:", attn_scores[0,0,0,:].max().item(), attn_scores[0,1,0,:].max().item())

        return out, attn_scores