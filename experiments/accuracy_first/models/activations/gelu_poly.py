from __future__ import annotations
import torch
import torch.nn as nn

class GeLUPoly2(nn.Module):
    """
    固定二次多项式近似：
      g(x) = a*x^2 + b*x + d
    默认系数按你给的 SOTA 近似初始化。
    """
    def __init__(self, a: float = 0.125, b: float = 0.25, d: float = 0.5):
        super().__init__()
        self.register_buffer("a", torch.tensor(a))
        self.register_buffer("b", torch.tensor(b))
        self.register_buffer("d", torch.tensor(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * (x * x) + self.b * x + self.d
