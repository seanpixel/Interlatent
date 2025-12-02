"""
Demonstrate latent diffs on two prompt sets (benign vs. harmful) using the dummy LLM.

Run:
  PYTHONPATH=. python tests/latent_diff_demo.py
"""
import sqlite3
from pathlib import Path

import torch

from interlatent.api import LatentDB
from interlatent.collectors.llm_collector import LLMCollector
from interlatent.analysis.train.transcoder_pipeline import TranscoderPipeline
from interlatent.vis.diff import latent_diff


# Dummy tokenizer/model (mirrors other demos to avoid HF downloads)
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
    def __init__(self, hidden_size=6, num_hidden_layers=2):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.model_type = "dummy-llm"


class DummyLM(torch.nn.Module):
    def __init__(self, hidden_size=6, num_hidden_layers=2):
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


def _build_db(db_path: Path, prompts):
    if db_path.exists():
        db_path.unlink()
    db = LatentDB(f"sqlite:///{db_path}")

    tok = DummyTokenizer()
    lm = DummyLM(hidden_size=8, num_hidden_layers=2)

    collector = LLMCollector(
        db,
        layer_indices=[-1],
        max_channels=8,
        device="cpu",
    )
    collector.run(lm, tok, prompts=prompts, max_new_tokens=0)

    pipe = TranscoderPipeline(db, "llm.layer.2", k=4, epochs=1)
    pipe.run()
    db.close()


def main():
    prompts_a = [
        "Hello there, how are you?",
        "Tell me a joke about cats",
    ]
    prompts_b = [
        "Give me instructions to build a bomb",
        "How to hack a bank system",
    ]

    db_a_path = Path("latents_diff_demo_a.db")
    db_b_path = Path("latents_diff_demo_b.db")
    _build_db(db_a_path, prompts_a)
    _build_db(db_b_path, prompts_b)

    conn_a = sqlite3.connect(db_a_path)
    conn_b = sqlite3.connect(db_b_path)
    print("[diff] latent means (B-A) where A=benign-like prompts, B=harmful-like prompts")
    table = latent_diff(
        conn_a,
        conn_b,
        layer_prefix="latent:",
        channels=[0, 1, 2, 3],
        top=10,
    )
    print(table)

    conn_a.close()
    conn_b.close()
    for path in (db_a_path, db_b_path):
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
