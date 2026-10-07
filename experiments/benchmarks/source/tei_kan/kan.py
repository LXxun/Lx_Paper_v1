"""Small fixed-grid B-spline KAN, implemented in ordinary PyTorch.

Each edge is a learned SiLU base term plus a cubic B-spline function.
This implementation is independent of the upstream KAN-MCP implementation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SplineKANLayer(nn.Module):
    def __init__(self, in_features, out_features, intervals=5, degree=3, bound=3.0):
        super().__init__()
        if intervals < 1 or degree < 1 or bound <= 0:
            raise ValueError("Invalid spline grid")
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        step = 2 * bound / intervals
        knots = (torch.arange(-degree, intervals + degree + 1).float() * step - bound)
        self.register_buffer("knots", knots)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, intervals + degree))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, std=0.03 / math.sqrt(in_features))

    def basis(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError("KAN input width mismatch")
        knots = self.knots.to(dtype=x.dtype)
        xx = x.unsqueeze(-1)
        b = ((xx >= knots[:-1]) & (xx < knots[1:])).to(x.dtype)
        for degree in range(1, self.degree + 1):
            left = (xx - knots[:-(degree + 1)]) / (knots[degree:-1] - knots[:-(degree + 1)])
            right = (knots[degree + 1:] - xx) / (knots[degree + 1:] - knots[1:-degree])
            b = left * b[..., :-1] + right * b[..., 1:]
        return b

    def forward(self, x):
        spline = F.linear(self.basis(x).flatten(-2), self.spline_weight.flatten(1))
        return F.linear(F.silu(x), self.base_weight, self.bias) + spline


class SmallKAN(nn.Module):
    def __init__(self, in_features, hidden=8, out_features=4, intervals=5, degree=3, bound=3.0):
        super().__init__()
        self.bound = bound
        self.layer1 = SplineKANLayer(in_features, hidden, intervals, degree, bound)
        self.layer2 = SplineKANLayer(hidden, out_features, intervals, degree, bound)

    def forward(self, x):
        h = self.layer1(x)
        # Keep hidden spline inputs inside their defined central grid as well.
        h = self.bound * torch.tanh(h / self.bound)
        return self.layer2(h)
