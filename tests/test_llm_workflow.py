import torch

from interlatent.api import LatentDB
from interlatent.collectors.llm_collector import LLMCollector
from interlatent.analysis.datasets import LinearProbeDataset
from interlatent.analysis.train.linear_probe_trainer import train_linear_probe
from interlatent.analysis.train.pipeline import TranscoderPipeline
from interlatent.analysis.train.sae_pipeline import SAEPipeline


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
            # make each layer distinct
            hidden_states.append(base + layer_idx)

        class Out:
            def __init__(self, hidden_states):
                self.hidden_states = tuple(hidden_states)

        return Out(hidden_states)


def test_llm_linear_probe_and_transcoder_and_sae(tmp_path):
    db = LatentDB(f"sqlite:///{tmp_path}/latents.db")
    tok = DummyTokenizer()
    lm = DummyLM(hidden_size=6, num_hidden_layers=2)

    def token_metrics_fn(prompt_idx, token_idx, token, **_):
        return {"token_id": token["id"], "token_pos": token_idx}

    collector = LLMCollector(
        db,
        layer_indices=[-1],
        max_channels=6,
        device="cpu",
        token_metrics_fn=token_metrics_fn,
    )
    collector.run(lm, tok, prompts=["a b", "a a c"], max_new_tokens=0)

    # Linear probe dataset builds and trains
    lp_ds = LinearProbeDataset(db, layer="llm.layer.2", target_key="token_id")
    probe = train_linear_probe(db, layer="llm.layer.2", target_key="token_id", epochs=2, lr=1e-2)
    assert len(lp_ds) > 0
    assert probe.proj.out_features == 1

    # Transcoder pipeline works on single-stream LLM activations
    pipe = TranscoderPipeline(db, "llm.layer.2", k=4, epochs=2)
    trainer = pipe.run()
    latent_events = db.fetch_activations(layer="latent:llm.layer.2")
    assert latent_events, "Expected latent activations from transcoder backfill"
    assert trainer.T.weight.shape[0] == 4

    # SAE pipeline produces latents too
    sae_pipe = SAEPipeline(db, "llm.layer.2", k=3, epochs=2)
    sae_model = sae_pipe.run()
    sae_latents = db.fetch_activations(layer="latent_sae:llm.layer.2")
    assert sae_latents, "Expected SAE latent activations"
    assert sae_model.encoder.out_features == 3

    db.close()
