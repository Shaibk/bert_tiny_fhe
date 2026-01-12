import torch
import torch.nn as nn

from .attention_plain import Attn2Quad, AttnPowerP2, AttnSoftmax
from .activations_plain import GeLUPoly2Learnable, GeLUPoly2Fixed

class PlainSelfAttention(nn.Module):
    def __init__(self, hidden, heads, dropout, attn_type="2quad", attn_kwargs=None):
        super().__init__()
        assert hidden % heads == 0
        self.hidden = hidden
        self.heads = heads
        self.d = hidden // heads
        attn_kwargs = attn_kwargs or {}

        # 关键：这些名字你后面要尽量对齐 FHE TinyBERT 的命名
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.wo  = nn.Linear(hidden, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

        if attn_type == "softmax":
            self.attn = AttnSoftmax(self.d, dropout=dropout)
        elif attn_type == "2quad":
            self.attn = Attn2Quad(self.d, dropout=dropout, **attn_kwargs)
        elif attn_type == "p2":
            self.attn = AttnPowerP2(self.d, dropout=dropout, **attn_kwargs)
        else:
            raise ValueError(attn_type)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B,T,3C]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.heads, self.d).transpose(1, 2)
        k = k.view(B, T, self.heads, self.d).transpose(1, 2)
        v = v.view(B, T, self.heads, self.d).transpose(1, 2)

        out, attn_prob = self.attn(q, k, v, attention_mask=attention_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.wo(out)
        out = self.drop(out)
        return out, attn_prob


class PlainFFN(nn.Module):
    def __init__(self, hidden, intermediate, dropout, act="gelu_poly_learnable", act_kwargs=None):
        super().__init__()
        act_kwargs = act_kwargs or {}
        self.fc1 = nn.Linear(hidden, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden)
        self.drop = nn.Dropout(dropout)

        if act == "gelu_poly_learnable":
            self.act = GeLUPoly2Learnable(**act_kwargs)
        elif act == "gelu_poly_fixed":
            self.act = GeLUPoly2Fixed(**act_kwargs)
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(act)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)   # 二次时只多一个平方
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PlainEncoderLayer(nn.Module):
    def __init__(self, hidden, heads, intermediate, dropout,
                 attn_type="2quad", attn_kwargs=None,
                 act="gelu_poly_learnable", act_kwargs=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.attn = PlainSelfAttention(hidden, heads, dropout, attn_type, attn_kwargs)
        self.ffn = PlainFFN(hidden, intermediate, dropout, act, act_kwargs)

    def forward(self, x, attention_mask=None):
        h = self.ln1(x)
        attn_out, attn_prob = self.attn(h, attention_mask=attention_mask)
        x = x + attn_out

        h2 = self.ln2(x)
        ffn_out = self.ffn(h2)
        x = x + ffn_out
        return x, attn_prob, h2


class PlainTinyBert(nn.Module):
    """
    明文 TinyBERT：只要你把 hidden/layers/heads/intermediate/vocab/max_len 对齐到 FHE 版本，
    导出的权重就能复用（或少量 key 映射后复用）。
    """
    def __init__(self, vocab_size, max_len,
                 hidden=128, layers=2, heads=2, intermediate=512,
                 dropout=0.1,
                 attn_type="2quad", attn_kwargs=None,
                 act="gelu_poly_learnable", act_kwargs=None,
                 num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Embedding(max_len, hidden)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            PlainEncoderLayer(hidden, heads, intermediate, dropout,
                              attn_type, attn_kwargs, act, act_kwargs)
            for _ in range(layers)
        ])

        # 任务 head：如果你做的是分类，就用这个
        self.cls = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)

        x = self.emb(input_ids) + self.pos(pos)
        x = self.drop(x)

        attn_probs, hiddens = [], []
        for layer in self.layers:
            x, ap, h2 = layer(x, attention_mask=attention_mask)
            attn_probs.append(ap)
            hiddens.append(h2)

        logits = self.cls(x[:, 0, :])  # [CLS]
        return {"logits": logits, "attn_probs": attn_probs, "hiddens": hiddens}
