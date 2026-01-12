import torch
import torch.nn as nn

class GeLUPoly2Fixed(nn.Module):
    def __init__(self, a=0.125, b=0.25, d=0.5):
        super().__init__()
        self.register_buffer("a", torch.tensor(float(a)))
        self.register_buffer("b", torch.tensor(float(b)))
        self.register_buffer("d", torch.tensor(float(d)))

    def forward(self, x):
        return self.a * (x * x) + self.b * x + self.d


class GeLUPoly2Learnable(nn.Module):
    def __init__(self, init_a=0.125, init_b=0.25, init_d=0.5):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(init_a)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))
        self.d = nn.Parameter(torch.tensor(float(init_d)))

    def forward(self, x):
        return self.a * (x * x) + self.b * x + self.d
