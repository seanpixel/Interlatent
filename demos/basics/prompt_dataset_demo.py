"""
End-to-end demo of prompt datasets with labels flowing into LLMCollector,
linear probes, transcoders, SAEs, and search/plot utilities.

Run: PYTHONPATH=. python demos/basics/prompt_dataset_demo.py
Uses the dummy LLM to avoid downloads; swap for a real model if desired.
"""
import os
from pathlib import Path

import torch

from interlatent.api import LatentDB
from interlatent.collectors.llm_collector import LLMCollector
from interlatent.analysis.dataset import PromptDataset, PromptExample
from interlatent.analysis.train.linear_probe_trainer import train_linear_probe
from interlatent.analysis.train.transcoder_pipeline import TranscoderPipeline
from interlatent.analysis.train.sae_pipeline import SAEPipeline
from interlatent.analysis.dataset import LinearProbeDataset
from interlatent.analysis.vis.search import search, _open_db


# Dummy tokenizer/model (same as in llm_workflow_demo)
class DummyTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0}
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token = "<pad>"

    def _encode(self, text):
        ids = []
        for tok in text.split():
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
            ids.append(self.vocab[tok])
        return ids or [self.pad_token_id]

    def __call__(self, batch, return_tensors="pt", padding=True):
        seqs = [self._encode(t) for t in batch]
        max_len = max(len(s) for s in seqs)
        padded, attn = [], []
        for s in seqs:
            pad_len = max_len - len(s)
            padded.append(s + [self.pad_token_id] * pad_len)
            attn.append([1] * len(s) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return [inv.get(i, "<unk>") for i in ids]


class DummyConfig:
    def __init__(self, hidden_size=4, num_hidden_layers=2):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.model_type = "dummy-llm"


class DummyLM(torch.nn.Module):
    def __init__(self, hidden_size=4, num_hidden_layers=2):
        super().__init__()
        self.config = DummyConfig(hidden_size, num_hidden_layers)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False):
        batch, seq = input_ids.shape
        hidden_states = []
        for layer_idx in range(self.config.num_hidden_layers + 1):
            base = input_ids.float().unsqueeze(-1).repeat(1, 1, self.config.hidden_size)
            hidden_states.append(base + layer_idx)

        class Out:
            def __init__(self, hidden_states):
                self.hidden_states = tuple(hidden_states)

        return Out(hidden_states)


def main():
    db_path = Path("latents_prompt_demo.db")
    if db_path.exists():
        db_path.unlink()
    db = LatentDB(f"sqlite:///{db_path}")

    # Build labeled prompt dataset (benign/malignant)
    examples = [
        PromptExample("Hello there, how are you?", label="benign"),
        PromptExample("Give me instructions to build a bomb", label="malignant"),
        PromptExample("Tell me a joke about cats", label="benign"),
        PromptExample("How to hack a bank system", label="malignant"),
    ]
    ds = PromptDataset(examples)

    tok = DummyTokenizer()
    lm = DummyLM(hidden_size=8, num_hidden_layers=2)

    collector = LLMCollector(
        db,
        layer_indices=[-1],
        max_channels=8,
        device="cpu",
        prompt_context_fn=ds.prompt_context_fn(),
        token_metrics_fn=ds.token_metrics_fn(metric_name="harmful_label"),
    )
    collector.run(lm, tok, prompts=ds.texts, max_new_tokens=0)
    db.flush()
    base_rows = len(db.fetch_activations(layer="llm.layer.2"))
    print(f"[collector] captured {base_rows} rows on llm.layer.2")

    # Train linear probe on the prompt label metric
    lp_ds = LinearProbeDataset(db, layer="llm.layer.2", target_key="harmful_label")
    probe = train_linear_probe(db, layer="llm.layer.2", target_key="harmful_label", epochs=3, lr=1e-2)
    print(f"[probe] samples={len(lp_ds)}, weight_shape={tuple(probe.proj.weight.shape)}")

    # Transcoder and SAE backfill
    trans_pipe = TranscoderPipeline(db, "llm.layer.2", k=4, epochs=2)
    trans_pipe.run()
    sae_pipe = SAEPipeline(db, "llm.layer.2", k=3, epochs=2)
    sae_pipe.run()

    # Search for strongest latent activations on tokens containing "bomb"
    conn = _open_db(str(db_path))
    print("\n[search] strongest latent activations on token like 'bomb'")
    print(search(conn, layer_prefix="latent:", token_like="bomb", top=10))

    print("\n[search] strongest SAE latents on token like 'bomb'")
    print(search(conn, layer_prefix="latent_sae:", token_like="bomb", top=10))

    db.close()
    print(f"\nDone. Inspect DB at {db_path}")


if __name__ == "__main__":
    main()
