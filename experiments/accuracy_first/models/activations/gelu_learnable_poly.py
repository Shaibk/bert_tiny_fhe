from __future__ import annotations
import torch
import torch.nn as nn

class GeLULearnablePoly2(nn.Module):
    """
    可学习二次多项式：g(x) = a*x^2 + b*x + d
    - 对齐FHE：仍是二次（只引入一次平方）
    - 训练更容易恢复精度（配合KD）
    """
    def __init__(self, init_a: float = 0.125, init_b: float = 0.25, init_d: float = 0.5):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(init_a)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))
        self.d = nn.Parameter(torch.tensor(float(init_d)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * (x * x) + self.b * x + self.d
