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

try:  # optional dependency
    from transformers import PreTrainedModel
except ImportError:  # pragma: no cover
    PreTrainedModel = None  # type: ignore

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
        ActivationEvent per (layer, channel, prompt, token).
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

        event_step = 0  # monotonically increasing step so PK uniqueness holds even with per-token events

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

            if max_new_tokens > 0 and hasattr(model, "generate") and hasattr(out, "sequences"):
                seq_tokens = out.sequences
                attn_mask_full = getattr(out, "attention_mask", None)
            else:
                seq_tokens = input_ids
                attn_mask_full = attn_mask

            if attn_mask_full is None:
                attn_mask_full = torch.ones_like(seq_tokens, device=seq_tokens.device)

            seq_tokens = seq_tokens.to(self.device)
            attn_mask_full = attn_mask_full.to(self.device)

            # Pre-decode tokens for readable context; truncate to real lengths later.
            tokens_decoded = [
                tokenizer.convert_ids_to_tokens(seq_tokens[b].tolist()) for b in range(seq_tokens.size(0))
            ]

            # hidden_states is a tuple(len = num_layers+1) of tensors (B, seq, hidden)
            for layer_idx in self._resolve_layers(len(hidden_states)):
                layer_tensor = hidden_states[layer_idx]  # (B, S, H)
                B, S, H = layer_tensor.shape
                if self.max_channels is not None:
                    H = min(H, self.max_channels)
                    layer_tensor = layer_tensor[:, :, :H]

                layer_name = f"llm.layer.{layer_idx}"

                for b_idx, prompt_text in enumerate(batch):
                    prompt_idx = i + b_idx
                    prompt_len = min(
                        int(attn_mask_full[b_idx].sum().item()),
                        layer_tensor.shape[1],
                        seq_tokens.shape[1],
                    )
                    seq_ids = seq_tokens[b_idx][:prompt_len].tolist()
                    seq_tokens_str = tokens_decoded[b_idx][:prompt_len]

                    for token_idx in range(prompt_len):
                        token_val = {
                            "id": seq_ids[token_idx],
                            "text": seq_tokens_str[token_idx] if token_idx < len(seq_tokens_str) else None,
                        }

                        for ch in range(H):
                            val = float(layer_tensor[b_idx, token_idx, ch].item())
                            ctx = {
                                "prompt_text": prompt_text,
                                "token_id": token_val["id"],
                                "layer_index": layer_idx,
                                "channel": ch,
                                "token_index": token_idx,
                                "prompt_index": prompt_idx,
                                "batch_offset": i,
                            }
                            self.db.write_event(
                                ActivationEvent(
                                    run_id=run_id,
                                    step=event_step,
                                    layer=layer_name,
                                    channel=ch,
                                    prompt=prompt_text,
                                    prompt_index=prompt_idx,
                                    token_index=token_idx,
                                    token=token_val["text"],
                                    tensor=[val],
                                    value_sum=val,
                                    value_sq_sum=val * val,
                                    context=ctx,
                                )
                            )
                            event_step += 1

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
        def _is_hf_model(obj) -> bool:
            if obj is None:
                return False
            if PreTrainedModel is not None and isinstance(obj, PreTrainedModel):
                return True
            return isinstance(obj, torch.nn.Module) and hasattr(obj, "config")

        def _follow(root, path: str):
            obj = root
            for attr in path.split("."):
                if obj is None:
                    return None
                obj = getattr(obj, attr, None)
            return obj

        # direct attribute
        direct = getattr(llm, "model", None)
        if _is_hf_model(direct):
            return direct
        if _is_hf_model(getattr(direct, "model", None)):
            return direct.model

        # vLLM engine path (0.4+)
        engine = getattr(llm, "llm_engine", None)
        try_paths = [
            "model_executor.driver_worker.model_runner.model",
            "model_executor.driver_worker.model_runner.driver_model",
            "model_executor.driver_worker._model_runner.model",
            "model_executor.driver_worker._model_runner.driver_model",
            # sometimes the HF model is nested under an extra .model
            "model_executor.driver_worker.model_runner.model.model",
        ]
        for path in try_paths:
            obj = _follow(engine, path)
            if _is_hf_model(obj):
                return obj
            nested = getattr(obj, "model", None)
            if _is_hf_model(nested):
                return nested

        # Fallback: shallow search for a torch.nn.Module with config.
        def _search(obj, depth: int = 0, max_depth: int = 3, visited=None):
            if obj is None or depth > max_depth:
                return None
            if visited is None:
                visited = set()
            if id(obj) in visited:
                return None
            visited.add(id(obj))
            if _is_hf_model(obj):
                return obj
            candidate_attrs = (
                "model",
                "model_runner",
                "driver_model",
                "_model_runner",
                "executor",
                "engine",
                "hf_model",
                "llm_model",
            )
            for attr in candidate_attrs:
                child = getattr(obj, attr, None)
                found = _search(child, depth + 1, max_depth, visited)
                if found is not None:
                    return found
            return None

        return _search(engine) or _search(llm)

    def _resolve_layers(self, num_hidden_states: int) -> Iterable[int]:
        """Normalize requested layer indices into valid positions."""
        for idx in self.layer_indices:
            if idx < 0:
                idx = num_hidden_states + idx
            if idx < 0 or idx >= num_hidden_states:
                raise IndexError(f"Layer index {idx} out of bounds for {num_hidden_states} hidden_states")
            yield idx
