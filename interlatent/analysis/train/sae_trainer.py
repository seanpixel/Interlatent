from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from interlatent.analysis.models.sae import SparseAutoencoder


class SAETrainer:
    """
    Minimal SAE trainer with L1 sparsity on the latent activation.
    """

    def __init__(self, in_dim: int, latent_dim: int):
        self.model = SparseAutoencoder(in_dim, latent_dim)

    def train(
        self,
        loader: DataLoader,
        *,
        epochs: int = 5,
        lr: float = 1e-3,
        l1: float = 1e-3,
        device: str | torch.device = "cpu",
    ):
        self.model.to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        mse = nn.MSELoss()

        for _ in range(epochs):
            for x, _ctx in loader:
                x = x.to(device)
                z, recon = self.model(x)
                loss = mse(recon, x) + l1 * z.abs().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.model.cpu()
        return self.model
