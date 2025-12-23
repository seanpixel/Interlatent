from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from interlatent.analysis.models.sae import SparseAutoencoder
from interlatent.utils.logging import get_logger

try:  # optional dependency
    from transformers import PreTrainedTokenizerBase
except ImportError:  # pragma: no cover
    PreTrainedTokenizerBase = object  # type: ignore

_LOG = get_logger(__name__)


@dataclass
class StreamingSAEConfig:
    layer_index: int = -1
    k: int = 128
    lr: float = 1e-3
    l1: float = 1e-3
    max_channels: int | None = None
    batch_size: int = 1
    max_new_tokens: int = 0
    sample_tokens: int | None = 2048  # sample this many tokens per batch for training
    device: str | torch.device | None = None


class StreamingSAETrainer:
    """
    Stream activations from an HF model to train an SAE online without storing them.
    """

    def __init__(self, config: StreamingSAEConfig):
        self.config = config
        self.model: SparseAutoencoder | None = None

    def _lazy_init(self, hidden_dim: int, device: torch.device):
        if self.model is None:
            self.model = SparseAutoencoder(hidden_dim, self.config.k)
            self.model.to(device)
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
            self.mse = nn.MSELoss()
            _LOG.info("Initialized SAE with in_dim=%d, k=%d", hidden_dim, self.config.k)

    def _train_step(self, x: torch.Tensor):
        z, recon = self.model(x)
        loss = self.mse(recon, x) + self.config.l1 * z.abs().mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item(), z.detach()

    def train(
        self,
        llm,
        tokenizer: PreTrainedTokenizerBase,
        prompts: Sequence[str],
    ):
        device = torch.device(self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        llm.eval().to(device)
        if self.model is not None:
            self.model.to(device)

        total_tokens = 0
        step = 0
        for i in range(0, len(prompts), self.config.batch_size):
            batch = prompts[i : i + self.config.batch_size]
            if tokenizer.pad_token_id is None and getattr(tokenizer, "eos_token_id", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            enc = tokenizer(batch, return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                out = llm(
                    **enc,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden_states = out.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden_states; ensure config.output_hidden_states=True")

            layer_idx = self.config.layer_index
            if layer_idx < 0:
                layer_idx = len(hidden_states) + layer_idx
            if layer_idx < 0 or layer_idx >= len(hidden_states):
                raise IndexError(f"layer_index {self.config.layer_index} out of range for {len(hidden_states)} states")

            hs = hidden_states[layer_idx]  # (B, S, H)
            if self.config.max_channels is not None:
                hs = hs[:, :, : self.config.max_channels]
            B, S, H = hs.shape
            mask = enc.get("attention_mask")
            if mask is None:
                mask = torch.ones((B, S), device=device, dtype=torch.bool)
            else:
                mask = mask.bool()
            flat = hs[mask]  # (N, H)
            if flat.numel() == 0:
                continue

            flat = flat.to(device=device, dtype=torch.float32)
            self._lazy_init(flat.shape[-1], device)

            if self.config.sample_tokens is not None and flat.shape[0] > self.config.sample_tokens:
                idx = torch.randperm(flat.shape[0], device=device)[: self.config.sample_tokens]
                flat = flat.index_select(0, idx)

            loss, z = self._train_step(flat)
            total_tokens += flat.shape[0]
            step += 1

            if step % 10 == 0:
                sparsity = (z > 0).float().sum(dim=1).mean().item()
                _LOG.info(
                    "StreamingSAE step %d | tokens %d | loss %.4f | nnz/token %.2f",
                    step,
                    total_tokens,
                    loss,
                    sparsity,
                )

        return self.model


class SAEFeatureWrapper(nn.Module):
    """
    Wrap a layer output to return sparse codes and optionally reconstructions.
    """

    def __init__(self, sae: SparseAutoencoder, return_reconstruction: bool = False):
        super().__init__()
        self.sae = sae
        self.return_reconstruction = return_reconstruction

    def forward(self, x):
        z, recon = self.sae(x)
        if self.return_reconstruction:
            return recon
        return z
