"""interlatent.llm.collector

Collect activations from language models served through vLLM (or the
underlying HF model) over a dataset of prompts.

Notes
-----
- vLLM does not currently expose hidden states through its public
  `LLM.generate` API. We therefore unwrap the underlying HF model and
  run forward passes with ``output_hidden_states=True`` to obtain
  per-layer activations.
- This is geared toward *small* models and small prompt sets. Capturing
  every hidden dimension for long sequences will explode storage; use
  `max_channels` to downsample.
"""
from __future__ import annotations

import uuid
from typing import Iterable, Sequence, Dict, Any

import torch

from interlatent.api.latent_db import LatentDB
from interlatent.schema import ActivationEvent, RunInfo
from interlatent.utils.logging import get_logger

try:  # optional dependency
    from transformers import PreTrainedTokenizerBase
except ImportError:  # pragma: no cover
    PreTrainedTokenizerBase = object  # type: ignore

_LOG = get_logger(__name__)

__all__ = ["VLLMCollector"]


class VLLMCollector:
    """
    Collect activations from a vLLM-served language model over a list of prompts.

    Parameters
    ----------
    db:
        LatentDB instance.
    layer_indices:
        Which hidden_states indices to log (0 = embeddings, 1 = first block, ...).
        Defaults to the final block only.
    max_channels:
        Optional cap on hidden dimensions to record (record first N).
    device:
        Torch device to run the HF model on (defaults to 'cuda' if available).
    """

    def __init__(
        self,
        db: LatentDB,
        *,
        layer_indices: Sequence[int] | None = None,
        max_channels: int | None = None,
        device: str | torch.device | None = None,
    ):
        self.db = db
        self.layer_indices = list(layer_indices) if layer_indices is not None else [-1]
        self.max_channels = max_channels
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ------------------------------------------------------------------
    def run(
        self,
        llm,
        tokenizer: PreTrainedTokenizerBase,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 0,
        batch_size: int = 1,
        tags: Dict[str, Any] | None = None,
    ) -> RunInfo:
        """
        Execute the underlying HF model on *prompts* and persist activations.

        This unwraps the vLLM object to grab its HF model, performs batched
        forwards with ``output_hidden_states=True``, and writes one
        ActivationEvent per (layer, channel) with the token-wise activations.
        """
        model = self._unwrap_hf_model(llm)
        if model is None:
            raise ValueError(
                "Could not locate HuggingFace model inside the provided vLLM object. "
                "Pass a vLLM LLM created in single-process mode or supply llm.model manually."
            )

        model.eval().to(self.device)

        run_id = uuid.uuid4().hex
        run_info = RunInfo(run_id=run_id, env_name=getattr(model.config, "model_type", "llm"), tags=tags or {})

        # simple batching over prompts
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"].to(self.device)
            attn_mask = enc.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(self.device)

            with torch.no_grad():
                # Generate completions if requested so activations include generated tokens.
                if max_new_tokens > 0 and hasattr(model, "generate"):
                    out = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=max_new_tokens,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        use_cache=False,
                    )
                    hidden_states = out.hidden_states
                    # HF returns a tuple over decode steps; take the last frame.
                    if hidden_states and isinstance(hidden_states[0], (list, tuple)):
                        hidden_states = hidden_states[-1]
                else:
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    hidden_states = out.hidden_states

            if hidden_states is None:
                raise RuntimeError("Model did not return hidden_states; ensure config.output_hidden_states=True")

            # hidden_states is a tuple(len = num_layers+1) of tensors (B, seq, hidden)
            for layer_idx in self._resolve_layers(len(hidden_states)):
                layer_tensor = hidden_states[layer_idx]  # (B, S, H)
                B, S, H = layer_tensor.shape
                if self.max_channels is not None:
                    H = min(H, self.max_channels)
                    layer_tensor = layer_tensor[:, :, :H]

                layer_name = f"llm.layer.{layer_idx}"

                # One event per hidden dimension; tensor = values across (batch, seq)
                for ch in range(H):
                    vals = layer_tensor[:, :, ch].reshape(-1).float().cpu()
                    ctx = {
                        "prompt_batch": batch,
                        "layer_index": layer_idx,
                        "channel": ch,
                        "token_count": S,
                        "batch_offset": i,
                    }
                    self.db.write_event(
                        ActivationEvent(
                            run_id=run_id,
                            step=i,  # batch index as step; per-channel differentiation via channel field
                            layer=layer_name,
                            channel=ch,
                            tensor=vals.tolist(),
                            value_sum=float(vals.sum()),
                            value_sq_sum=float((vals * vals).sum()),
                            context=ctx,
                        )
                    )

        self.db.flush()
        _LOG.info("vLLM collection finished: %s (%d prompts)", run_id, len(prompts))
        return run_info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _unwrap_hf_model(self, llm):
        """
        Try a handful of known attribute paths to pull the HF model out of a vLLM LLM.

        Works in single-process CPU/GPU setups; multi-node setups might keep the model
        in a worker process and are out of scope for now.
        """
        # direct attribute
        if hasattr(llm, "model"):
            return llm.model
        # vLLM engine path (0.4+)
        engine = getattr(llm, "llm_engine", None)
        try_paths = [
            "model_executor.driver_worker.model_runner.model",
            "model_executor.driver_worker.model_runner.driver_model",
        ]
        for path in try_paths:
            obj = engine
            for attr in path.split("."):
                if obj is None:
                    break
                obj = getattr(obj, attr, None)
            if obj is not None:
                return obj
        return None

    def _resolve_layers(self, num_hidden_states: int) -> Iterable[int]:
        """Normalize requested layer indices into valid positions."""
        for idx in self.layer_indices:
            if idx < 0:
                idx = num_hidden_states + idx
            if idx < 0 or idx >= num_hidden_states:
                raise IndexError(f"Layer index {idx} out of bounds for {num_hidden_states} hidden_states")
            yield idx
