from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention.softmax_attention import SoftmaxAttention
from .attention.quad_attention import QuadAttention
from .attention.power_softmax_attention import PowerSoftmaxAttention
from .activations.gelu_poly import GeLUPoly2
from .activations.gelu_learnable_poly import GeLULearnablePoly2

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float,
                 attn_type: str, attn_kwargs: dict):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=True)
        self.drop = nn.Dropout(dropout)

        if attn_type == "softmax":
            self.attn = SoftmaxAttention(self.head_dim, dropout=dropout)
        elif attn_type == "quad":
            self.attn = QuadAttention(self.head_dim, dropout=dropout, **attn_kwargs)
        elif attn_type == "power":
            self.attn = PowerSoftmaxAttention(self.head_dim, dropout=dropout, **attn_kwargs)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

    def forward(self, x: torch.Tensor, attn_mask=None):
        # x: [B,T,C]
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B,T,3C]  (严格口径：+1乘法层)
        q, k, v = qkv.chunk(3, dim=-1)
        # [B,H,T,Dh]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        out, prob = self.attn(q, k, v, attn_mask=attn_mask)  # QK^T(+1) + approx(+1) + PV(+1)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.wo(out)  # Wo (+1)
        out = self.drop(out)
        return out, prob

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int,
                 dropout: float, attn_type: str, attn_kwargs: dict,
                 act_type: str, act_kwargs: dict):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.mha = MultiHeadSelfAttention(hidden_size, num_heads, dropout, attn_type, attn_kwargs)

        self.ffn1 = nn.Linear(hidden_size, intermediate_size)  # FFN1 (+1)
        self.ffn2 = nn.Linear(intermediate_size, hidden_size)  # FFN2 (+1)
        self.drop = nn.Dropout(dropout)

        if act_type == "gelu":
            self.act = nn.GELU()
        elif act_type == "gelu_poly_fixed":
            self.act = GeLUPoly2(**act_kwargs)
        elif act_type == "gelu_poly_learnable":
            self.act = GeLULearnablePoly2(**act_kwargs)
        else:
            raise ValueError(f"Unknown act_type: {act_type}")

    def forward(self, x: torch.Tensor, attn_mask=None):
        # Pre-LN
        h = self.ln1(x)
        attn_out, attn_prob = self.mha(h, attn_mask=attn_mask)
        x = x + attn_out

        h2 = self.ln2(x)
        y = self.ffn1(h2)
        y = self.act(y)  # 二次多项式时：平方 (+1)
        y = self.drop(y)
        y = self.ffn2(y)
        y = self.drop(y)
        x = x + y
        return x, attn_prob, h2  # 返回h2用于hidden蒸馏（可选）

class BertTinyVariant(nn.Module):
    """
    一个最小可跑的BERT-Tiny Encoder骨架。
    你需要把 embedding / tokenizer / pooler / head 对齐你们工程现有实现。
    这里提供“可插拔注意力/激活 + 蒸馏信号输出”。
    """
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int,
                 num_heads: int, intermediate_size: int, dropout: float,
                 attn_type: str, attn_kwargs: dict, act_type: str, act_kwargs: dict,
                 num_classes: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.pos = nn.Embedding(512, hidden_size)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, intermediate_size, dropout,
                                    attn_type, attn_kwargs, act_type, act_kwargs)
            for _ in range(num_layers)
        ])

        self.cls = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attn_mask=None):
        # input_ids: [B,T]
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.emb(input_ids) + self.pos(pos)
        x = self.drop(x)

        attn_probs = []
        hiddens = []
        for layer in self.layers:
            x, attn_prob, hidden_for_kd = layer(x, attn_mask=attn_mask)
            attn_probs.append(attn_prob)
            hiddens.append(hidden_for_kd)

        # 简化：用 [CLS] 位置作为分类
        logits = self.cls(x[:, 0, :])
        return {
            "logits": logits,
            "attn_probs": attn_probs,
            "hiddens": hiddens,
        }
