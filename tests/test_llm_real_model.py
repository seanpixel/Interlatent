"""
Slow, optional integration test using a real HF model (default: SmolLM-360M).

Enable by setting RUN_LLM_REAL=1 to avoid pulling weights during normal CI.
"""
import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interlatent.api import LatentDB
from interlatent.collectors.llm_collector import LLMCollector
from interlatent.analysis.datasets import LinearProbeDataset
from interlatent.analysis.train.linear_probe_trainer import train_linear_probe
from interlatent.analysis.train.pipeline import TranscoderPipeline
from interlatent.analysis.train.sae_pipeline import SAEPipeline


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LLM_REAL") != "1",
    reason="Set RUN_LLM_REAL=1 to run real-model LLM integration test (downloads weights).",
)


def test_llm_real_model_end_to_end(tmp_path):
    model_id = os.environ.get("LLM_MODEL", "HuggingFaceTB/SmolLM-360M")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    db = LatentDB(f"sqlite:///{tmp_path}/latents.db")

    def token_metrics_fn(prompt_idx, token_idx, token, **_):
        return {"token_id": token["id"], "token_pos": token_idx}

    collector = LLMCollector(
        db,
        layer_indices=[-1],
        max_channels=64,
        device=device,
        token_metrics_fn=token_metrics_fn,
    )
    collector.run(
        llm,
        tok,
        prompts=["Hello world", "Why is the sky blue?" ],
        max_new_tokens=0,
        batch_size=1,
    )

    # Linear probe
    lp_ds = LinearProbeDataset(db, layer="llm.layer.-1", target_key="token_id")
    probe = train_linear_probe(db, layer="llm.layer.-1", target_key="token_id", epochs=1, lr=1e-3, batch_size=16)
    assert len(lp_ds) > 0
    assert probe.proj.out_features == 1

    # Transcoder
    pipe = TranscoderPipeline(db, "llm.layer.-1", k=8, epochs=1)
    pipe.run()
    latent_events = db.fetch_activations(layer="latent:llm.layer.-1")
    assert latent_events, "Expected latent activations from transcoder backfill"

    # SAE
    sae_pipe = SAEPipeline(db, "llm.layer.-1", k=8, epochs=1)
    sae_pipe.run()
    sae_latents = db.fetch_activations(layer="latent_sae:llm.layer.-1")
    assert sae_latents, "Expected SAE latent activations"

    db.close()
