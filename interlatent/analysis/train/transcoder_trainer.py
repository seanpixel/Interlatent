from __future__ import annotations

import torch
import torch.nn as nn


class TranscoderTrainer:
    def __init__(self, in_dim, out_dim, k_latent):
        self.T = nn.Linear(in_dim, k_latent, bias=False)   # encoder
        self.R = nn.Linear(k_latent, out_dim, bias=False)  # decoder

    def train(self, loader, *, epochs=5, lr=1e-3, l1=1e-3, device="cpu"):
        self.T.to(device); self.R.to(device)
        opt = torch.optim.Adam(list(self.T.parameters()) + list(self.R.parameters()), lr=lr)

        for _ in range(epochs):
            for x_pre, x_post in loader:
                x_pre, x_post = x_pre.to(device), x_post.to(device)
                z      = self.T(x_pre)
                recon  = self.R(z)
                mse    = ((recon - x_post) ** 2).mean()
                spars  = z.abs().mean()
                loss   = mse + l1 * spars
                opt.zero_grad(); loss.backward(); opt.step()
        return {
            "encoder": self.T.cpu().state_dict(),
            "decoder": self.R.cpu().state_dict(),
        }
