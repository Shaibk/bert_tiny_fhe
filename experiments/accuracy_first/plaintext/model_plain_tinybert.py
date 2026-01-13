import torch
import torch.nn as nn

# 引入底层的数学实现
from .attention_plain import Attn2Quad, SoftmaxAttention

class BiasLayer(nn.Module):
    """
    FHE 深度优化层：只做加法，不消耗 Level。
    y = x + bias
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return x + self.bias

class PlainSelfAttention(nn.Module):
    """
    封装了 Q, K, V 投影层的 Attention 模块
    """
    def __init__(self, hidden, heads, dropout, attn_type="2quad", attn_kwargs=None, learnable_tau=True):
        super().__init__()
        assert hidden % heads == 0
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        attn_kwargs = attn_kwargs or {}

        # Q, K, V, Output 投影层 (FHE 优化: bias=False)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.wo  = nn.Linear(hidden, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

        # Tau 参数
        if learnable_tau:
            # 配合 static_scale=0.01，初始 tau=5.0 使得总缩放 ~1.0
            self.tau = nn.Parameter(torch.full((heads,), 5.0))
        else:
            self.register_buffer("tau", torch.full((heads,), 5.0))

        # 实例化具体的数学计算核心
        if attn_type == "softmax":
            # Softmax 不需要 head_dim 做参数(内部用)，也不需要 c
            self.attn = SoftmaxAttention(self.head_dim, dropout=dropout)
        elif attn_type == "2quad":
            # [Fix] 这里正确传递参数
            # head_dim 是位置参数，其他通过 kwargs 传递 (包含 c)
            self.attn = Attn2Quad(self.head_dim, dropout=dropout, **attn_kwargs)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

    def forward(self, x, mask=None):
        B, T, C = x.shape
        # 1. 投影
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape heads
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # 3. 计算 Attention (传入 tau)
        # 注意：Attn2Quad.forward 签名需要支持 tau
        out, _ = self.attn(q, k, v, attention_mask=mask, tau=self.tau)

        # 4. 还原形状 & 输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.wo(out)
        out = self.drop(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, hidden, heads, intermediate, dropout, attn_type, attn_kwargs, act, act_kwargs, norm_type="bias_only", learnable_tau=True):
        super().__init__()
        
        # 1. Self Attention (使用封装好的类)
        self.attention = PlainSelfAttention(
            hidden, heads, dropout, 
            attn_type=attn_type, 
            attn_kwargs=attn_kwargs,
            learnable_tau=learnable_tau
        )
        
        # 2. FFN
        from .activations_plain import get_activation
        self.linear1 = nn.Linear(hidden, intermediate)
        self.activation = get_activation(act, **act_kwargs)
        self.linear2 = nn.Linear(intermediate, hidden)
        self.dropout = nn.Dropout(dropout)

        # 3. Norm Strategy
        self.norm_type = norm_type
        if norm_type == "bn":
            self.norm1 = nn.BatchNorm1d(hidden)
            self.norm2 = nn.BatchNorm1d(hidden)
        elif norm_type == "bias_only":
            self.norm1 = BiasLayer(hidden)
            self.norm2 = BiasLayer(hidden)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x, mask):
        # Attention Sub-layer
        residual = x
        out = self.attention(x, mask)
        x = residual + 0.2*out
        
        if self.norm_type == "bn":
            x = x.transpose(1, 2)
            x = self.norm1(x)
            x = x.transpose(1, 2)
        else:
            x = self.norm1(x)

        # FFN Sub-layer
        residual = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.dropout(out)
        x = residual + out
        
        if self.norm_type == "bn":
            x = x.transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2)
        else:
            x = self.norm2(x)
            
        return x

class PlainTinyBert(nn.Module):
    def __init__(self, vocab_size, max_len, hidden, layers, heads, intermediate, dropout, 
                 attn_type, attn_kwargs, act, act_kwargs, 
                 norm_type="bias_only", learnable_tau=True, num_classes=2, **kwargs):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, hidden))
        
        # Embedding Norm
        if norm_type == "bn":
            self.emb_norm = nn.LayerNorm(hidden) 
        else:
            self.emb_norm = nn.Identity()

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden, heads, intermediate, dropout, 
                attn_type, attn_kwargs, act, act_kwargs, 
                norm_type, learnable_tau
            )
            for _ in range(layers)
        ])
        
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        
        x = self.embedding(input_ids) + self.pos_embedding[:, :T, :]
        x = self.emb_norm(x)

        for layer in self.layers:
            x = layer(x, attention_mask)
        
        logits = self.classifier(x[:, 0, :])
        return {"logits": logits}