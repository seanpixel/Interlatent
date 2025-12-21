"""
Forward-hook based patching to inject latent-based deltas into a model layer.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import torch
import torch.nn as nn

from .latents import LatentBasis


MaskFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
# mask_fn(hidden_states, attention_mask) -> mask of shape (B, S, 1)


def _default_mask_fn(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    # Apply to all tokens; broadcast to (B, S, 1)
    B, S, _ = hidden_states.shape
    return torch.ones((B, S, 1), device=hidden_states.device, dtype=hidden_states.dtype)


def prompt_only_mask(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """
    Apply only to the prompt tokens (i.e., tokens present in attention_mask).
    For generated tokens, attention_mask often grows, but we treat mask>0 as prompt.
    """
    if attention_mask is None:
        return _default_mask_fn(hidden_states, attention_mask)
    mask = (attention_mask > 0).unsqueeze(-1).to(hidden_states.dtype)
    return mask


@dataclass
class InterventionConfig:
    basis: LatentBasis
    channels: Sequence[int]
    scales: Sequence[float]
    mask_fn: MaskFn = _default_mask_fn

    def composed_vector(self) -> torch.Tensor:
        return self.basis.composed_vector(self.channels, self.scales)


@contextlib.contextmanager
def patch_layer(
    model: nn.Module,
    layer_module: nn.Module,
    config: InterventionConfig,
):
    """
    Context manager that registers a forward hook on `layer_module` adding the
    composed latent vector to hidden states (broadcast across tokens, masked).
    """
    delta = config.composed_vector()
    if delta.ndim != 1:
        raise ValueError("Intervention vector must be 1D (hidden_dim,)")
    warned = False

    def _hook(_mod, inputs, output):
        # output: hidden_states tensor
        if not isinstance(output, torch.Tensor):
            return output
        hs = output
        mask = config.mask_fn(hs, getattr(inputs[0], "attention_mask", None) if inputs else None)
        if mask.shape[:2] != hs.shape[:2]:
            # Try to fall back if mask_fn didn't use inputs properly.
            mask = torch.ones_like(hs[..., :1])
        d = delta
        if d.shape[0] != hs.shape[-1]:
            nonlocal warned
            if not warned:
                print(
                    f"[intervention] Mismatched hidden dim: delta {d.shape[0]} vs layer {hs.shape[-1]}; "
                    "padding/truncating to match."
                )
                warned = True
            if d.shape[0] < hs.shape[-1]:
                pad = torch.zeros(hs.shape[-1] - d.shape[0], device=d.device, dtype=d.dtype)
                d = torch.cat([d, pad], dim=0)
            else:
                d = d[: hs.shape[-1]]
        d = d.to(hs.device, dtype=hs.dtype)
        return hs + mask * d

    handle = layer_module.register_forward_hook(_hook)
    try:
        yield model
    finally:
        handle.remove()
