"""
Smoke test for Ministral-3-14B with text-only prompts plus latent collection
and small Transcoder/SAE training (mirrors llm_real_model_demo).
Run manually: RUN_MINISTRAL3=1 PYTHONPATH=. python scripts/demos/ministral3_demo.py
Requires: transformers>=4.57, mistral_common, and GPU recommended. Quantization
paths expect torch builds exposing torch.nn.Module.set_submodule; a shim is
installed here for older torch versions.
"""
import os

from pathlib import Path

import torch

from interlatent.api import LatentDB
from interlatent.collectors.llm_collector import LLMCollector
from interlatent.analysis.datasets import LinearProbeDataset
from interlatent.analysis.train.linear_probe_trainer import train_linear_probe
from interlatent.analysis.train.transcoder_pipeline import TranscoderPipeline
from interlatent.analysis.train.sae_pipeline import SAEPipeline


def _ensure_set_submodule():
    """
    Fine-grained quantization in Transformers calls torch.nn.Module.set_submodule.
    Older torch releases omit it; add a minimal version so replacement works.
    """
    if hasattr(torch.nn.Module, "set_submodule"):
        return

    def set_submodule(self, target, module):  # type: ignore[override]
        if not target:
            raise AttributeError("set_submodule: empty target")
        atoms = target.split(".")
        parent = self
        for name in atoms[:-1]:
            parent = getattr(parent, name)
        setattr(parent, atoms[-1], module)
        return self

    torch.nn.Module.set_submodule = set_submodule  # type: ignore[attr-defined]


def main():
    if os.environ.get("RUN_MINISTRAL3") != "1":
        print("Set RUN_MINISTRAL3=1 to run (downloads weights); skipping.")
        return

    from transformers import AutoConfig, Mistral3ForConditionalGeneration, MistralCommonBackend

    model_id = os.environ.get("LLM_MODEL", "mistralai/Ministral-3-14B-Instruct-2512")
    trust_remote_code = os.environ.get("HF_TRUST_REMOTE_CODE", "1") == "1"

    _ensure_set_submodule()

    print(f"Loading tokenizer for {model_id} ...")
    tok = MistralCommonBackend.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model on {device} ...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    llm = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        config=config,
        torch_dtype=dtype,
        device_map={"": device},
    )

    prompts = [
        "Explain why the sky is blue in one paragraph.",
        "List three creative uses for a paperclip.",
    ]

    print("Generating ...")
    tok_inputs = tok(prompts, return_tensors="pt", padding=True)
    for k, v in tok_inputs.items():
        if torch.is_tensor(v):
            tok_inputs[k] = v.to(device=device)
    gen_out = llm.generate(**tok_inputs, max_new_tokens=64)
    decoded = tok.batch_decode(gen_out, skip_special_tokens=True)
    for i, text in enumerate(decoded, 1):
        print(f"\n--- Sample {i} ---\n{text}")

    db_path = Path("latents_ministral3.db")
    if db_path.exists():
        db_path.unlink()
    db = LatentDB(f"sqlite:///{db_path}")

    def token_metrics_fn(prompt_idx, token_idx, token, **_):
        return {"token_id": token["id"], "token_pos": token_idx}

    collector = LLMCollector(
        db,
        layer_indices=[20],
        max_channels=128,
        device=device,
        token_metrics_fn=token_metrics_fn,
    )
    collector.run(llm, tok, prompts=prompts, max_new_tokens=0, batch_size=1)
    base_rows = len(db.fetch_activations(layer="llm.layer.20"))
    print(f"[collector] captured {base_rows} activations for layer llm.layer.20")

    lp_ds = LinearProbeDataset(db, layer="llm.layer.20", target_key="token_id")
    probe = train_linear_probe(
        db, layer="llm.layer.20", target_key="token_id", epochs=1, lr=1e-3, batch_size=16
    )
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
    print("Done. Inspect latents in latents_ministral3.db")


if __name__ == "__main__":
    main()
