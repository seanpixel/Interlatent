"""interlatent.hooks

Torch-specific forward-hook utilities.

`TorchHook` is a *context manager*; inside the `with` block, every forward
pass through the specified layers emits `ActivationEvent`s into a
`LatentDB`.  Leave the context and all hooks auto‑deregister, avoiding
reference cycles.

*Assumptions* (v0):
• Activations are `torch.Tensor`s shaped *(B, C, …)* where dimension 1 is
  the channel index.  We flatten spatial dims per channel.  Works for
  linear layers too because spatial dims = 0.
• Batch size may vary. Each sample in the batch gets its own event with
  consecutive `step` numbers local to the run.
"""
from __future__ import annotations

import torch

import itertools
import weakref
from contextlib import AbstractContextManager, ExitStack
from typing import Dict, List, Sequence, Callable, Any, Optional

from .api.latent_db import LatentDB
from .schema import ActivationEvent
from .utils.logging import get_logger

_LOG = get_logger(__name__)

__all__ = ["TorchHook"]


class TorchHook(AbstractContextManager):
    """Register forward hooks that push activations into LatentDB."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        context_supplier: Optional[Callable[[], Dict[str, Any]]] = None,
        layers: Sequence[str],
        db: LatentDB,
        run_id: str,
        max_channels: int | None = None,  # keep memory sane for huge convs
    ) -> None:
        self.model = model
        self.layers = list(layers)
        self.db = db
        self.run_id = run_id
        self.max_channels = max_channels
        self._ctx_fn = context_supplier or (lambda: {})

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._step_counter = itertools.count().__next__  # atomic-ish counter

        # Map layer names to modules; allow attribute dotted paths.
        self._module_lookup: Dict[str, torch.nn.Module] = {}
        for name in self.layers:
            mod = self._find_submodule(model, name)
            if mod is None:
                raise ValueError(f"Layer '{name}' not found in model")
            self._module_lookup[name] = mod

    # ------------------------------------------------------------------
    # Context manager protocol -----------------------------------------
    # ------------------------------------------------------------------

    def __enter__(self):
        for layer_name, module in self._module_lookup.items():
            handle = module.register_forward_hook(self._make_hook(layer_name))
            self._handles.append(handle)
        _LOG.debug("TorchHook registered %d hooks", len(self._handles))
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        for h in self._handles:
            h.remove()
        self._handles.clear()
        _LOG.debug("TorchHook removed hooks")
        return False  # propagate exceptions

    # ------------------------------------------------------------------
    # Internal helpers --------------------------------------------------
    # ------------------------------------------------------------------

    def _make_hook(self, layer_name: str):
        """Factory returning the closure used as forward_hook."""

        db_ref = weakref.ref(self.db)
        run_id = self.run_id
        max_channels = self.max_channels
        step_counter = self._step_counter

        def _hook(module, inp, out):  # noqa: D401 – PyTorch hook signature
            db = db_ref()
            if db is None:
                return  # LatentDB GC'ed? should not happen.

            tensor = out.detach().cpu()
            if tensor.ndim < 2:
                tensor = tensor.unsqueeze(1)  # (B,1)
            B, C = tensor.shape[:2]
            if max_channels is not None:
                C = min(C, max_channels)
                tensor = tensor[:, :C]

            # Flatten spatial dims per channel
            tensor = tensor.reshape(B, C, -1)

            for b in range(B):
                step = step_counter()
                for ch in range(C):
                    vals = tensor[b, ch].float().view(-1)
                    ctx = dict(self._ctx_fn())
                    ev = ActivationEvent(
                        run_id=run_id,
                        step=step,
                        layer=layer_name,
                        channel=ch,
                        tensor=vals.tolist(),
                        value_sum=float(vals.sum()),
                        value_sq_sum=float((vals**2).sum()),
                        context=ctx,  
                    )
                    db.write_event(ev)

        return _hook

    @staticmethod
    def _find_submodule(root: torch.nn.Module, dotted: str):
        mod = root
        for attr in dotted.split("."):
            mod = getattr(mod, attr, None)
            if mod is None:
                return None
        return mod


class PrePostHookCtx:
    """
    Context manager that registers both pre- and post-forward hooks for the
    requested layers, streaming to a LatentDB.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layers: list[str],
        *,
        db: LatentDB,
        run_id: str,
        context_supplier,             # lambda -> dict containing 'step' + misc
        device: torch.device | str = "cpu",
    ):
        self._model = model
        self._layers = layers
        self._db = db
        self._run_id = run_id
        self._ctx = context_supplier
        self._device = torch.device(device)
        self._stack = ExitStack()

    # ------------------------------------------------------------------ enter/exit
    def __enter__(self):
        for name in self._layers:
            mod = self._get_submodule(name)
            if mod is None:
                raise ValueError(f"Layer '{name}' not found in model")

            # pre-forward
            self._stack.enter_context(
                mod.register_forward_pre_hook(self._make_cb(name, which="pre"))
            )
            # post-forward
            self._stack.enter_context(
                mod.register_forward_hook(self._make_cb(name, which="post"))
            )
        return self

    def __exit__(self, *exc):
        self._stack.close()
        return False

    # ------------------------------------------------------------------ helpers
    def _get_submodule(self, dotted):
        obj = self._model
        for part in dotted.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    def _make_cb(self, layer_name: str, *, which: str):
        tag = f"{layer_name}:{which}"

        def _record(tensor):
            tensor = tensor.detach().to("cpu")
            if tensor.dim() == 2:            # (B, C)
                for ch, col in enumerate(tensor.squeeze(0)):
                    self._write(tag, ch, col)
            else:                            # flatten everything
                flat = tensor.view(-1)
                for idx, val in enumerate(flat):
                    self._write(tag, idx, val)

        if which == "pre":
            # pre signature: (module, inp)
            def _cb(module, inp):
                _record(inp[0])
            return _cb
        else:
            # post signature: (module, inp, out)
            def _cb(module, inp, out):
                _record(out)
            return _cb

    def _write(self, layer_tag, channel, val):
        ctx = self._ctx() or {}
        step = ctx.get("step", 0)
        self._db.write_event(
            ActivationEvent(
                run_id=self._run_id,
                step=step,
                layer=layer_tag,
                channel=channel,
                tensor=[float(val)],
                context=ctx,
                value_sum=float(val),
                value_sq_sum=float(val * val),
            )
        )
