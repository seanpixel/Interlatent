from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Simple sparse autoencoder with ReLU bottleneck and L1 penalty applied
    to the latent activations during training.
    """

    def __init__(self, in_dim: int, latent_dim: int, *, bias: bool = False):
        super().__init__()
        self.encoder = nn.Linear(in_dim, latent_dim, bias=bias)
        self.decoder = nn.Linear(latent_dim, in_dim, bias=bias)

    def forward(self, x):
        z = F.relu(self.encoder(x))
        recon = self.decoder(z)
        return z, recon

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearTranscoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Linear(in_dim,  latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, out_dim,  bias=False)

    def forward(self, x):
        z   = F.relu(self.encoder(x))        # latent bottleneck
        out = self.decoder(z)
        return z, out
