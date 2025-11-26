"""
Interactive demo with a real HF model (default SmolLM-360M).
Run manually: RUN_LLM_REAL=1 PYTHONPATH=. python tests/llm_real_model_demo.py
Downloads weights; keep off in CI.
"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interlatent.api import LatentDB
from interlatent.collectors.llm_collector import LLMCollector
from interlatent.analysis.datasets import LinearProbeDataset
from interlatent.analysis.train.linear_probe_trainer import train_linear_probe
from interlatent.analysis.train.pipeline import TranscoderPipeline
from interlatent.analysis.train.sae_pipeline import SAEPipeline


def main():
    if os.environ.get("RUN_LLM_REAL") != "1":
        print("Set RUN_LLM_REAL=1 to run (downloads weights); skipping.")
        return

    model_id = os.environ.get("LLM_MODEL", "HuggingFaceTB/SmolLM-360M")
    print(f"Loading model {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    db = LatentDB("sqlite:///latents_llm_real.db")

    def token_metrics_fn(prompt_idx, token_idx, token, **_):
        return {"token_id": token["id"], "token_pos": token_idx}

    collector = LLMCollector(
        db,
        layer_indices=[20],
        max_channels=128,
        device=device,
        token_metrics_fn=token_metrics_fn,
    )
    prompts = ["Hello world", "Why is the sky blue?"]
    collector.run(llm, tok, prompts=prompts, max_new_tokens=0, batch_size=1)
    base_rows = len(db.fetch_activations(layer="llm.layer.20"))
    print(f"[collector] captured {base_rows} activations for layer llm.layer.20")

    lp_ds = LinearProbeDataset(db, layer="llm.layer.20", target_key="token_id")
    probe = train_linear_probe(db, layer="llm.layer.20", target_key="token_id", epochs=1, lr=1e-3, batch_size=16)
    print(f"[linear probe] samples={len(lp_ds)}, weight_shape={tuple(probe.proj.weight.shape)}")

    pipe = TranscoderPipeline(db, "llm.layer.20", k=8, epochs=1)
    trainer = pipe.run()
    latent_events = db.fetch_activations(layer="latent:llm.layer.20")
    print(f"[transcoder] latent rows={len(latent_events)}, encoder_shape={tuple(trainer.T.weight.shape)}")

    sae_pipe = SAEPipeline(db, "llm.layer.20", k=8, epochs=1)
    sae_model = sae_pipe.run()
    sae_latents = db.fetch_activations(layer="latent_sae:llm.layer.20")
    print(f"[sae] latent rows={len(sae_latents)}, encoder_shape={tuple(sae_model.encoder.weight.shape)}")

    db.close()
    print("Done. Inspect latents in latents_llm_real.db")


if __name__ == "__main__":
    main()
