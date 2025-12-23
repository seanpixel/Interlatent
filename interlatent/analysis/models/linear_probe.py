from __future__ import annotations

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """
    Minimal linear probe for regression or classification tasks on activations.
    """

    def __init__(self, in_dim: int, out_dim: int = 1, *, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


