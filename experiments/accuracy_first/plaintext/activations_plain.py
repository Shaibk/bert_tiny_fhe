import torch
import torch.nn as nn
import torch.nn.functional as F

class GeLUPoly2Learnable(nn.Module):
    """
    可学习的二次多项式激活函数: y = a*x^2 + b*x + d
    用于模拟 GELU，同时保持 FHE 友好的低深度特性。
    """
    def __init__(self, init_a=0.044715, init_b=0.5, init_d=0.0):
        super().__init__()
        # 定义为可学习参数
        self.a = nn.Parameter(torch.tensor(float(init_a)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))
        self.d = nn.Parameter(torch.tensor(float(init_d)))

    def forward(self, x):
        return self.a * (x**2) + self.b * x + self.d

class GeLUPoly2Fixed(nn.Module):
    """
    固定系数的二次多项式激活函数 (不学习)
    """
    def __init__(self, a=0.044715, b=0.5, d=0.0):
        super().__init__()
        self.register_buffer('a', torch.tensor(float(a)))
        self.register_buffer('b', torch.tensor(float(b)))
        self.register_buffer('d', torch.tensor(float(d)))

    def forward(self, x):
        return self.a * (x**2) + self.b * x + self.d

def get_activation(act_type, **kwargs):
    """
    激活函数工厂
    """
    if act_type == "gelu":
        return nn.GELU()
    elif act_type == "relu":
        return nn.ReLU()
    elif act_type == "gelu_poly_learnable":
        return GeLUPoly2Learnable(**kwargs)
    elif act_type == "gelu_poly_fixed":
        # 兼容参数名差异 (init_a -> a)
        clean_kwargs = {}
        if 'init_a' in kwargs: clean_kwargs['a'] = kwargs['init_a']
        if 'init_b' in kwargs: clean_kwargs['b'] = kwargs['init_b']
        if 'init_d' in kwargs: clean_kwargs['d'] = kwargs['init_d']
        return GeLUPoly2Fixed(**clean_kwargs)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")