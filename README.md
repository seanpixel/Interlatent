# Interlatent

Interlatent is a lightweight interpretability toolkit where you can: save prompts and activations with context, attach labels, learn sparse latents (transcoders/SAEs) and probes, and quickly see which tokens or states drive them. The goal is to allow new independent researchers / engineers to dabble with understanding their models. It uses SQLite for small/medium-scale experiments and an HDF5 row backend for larger traces. We are still in development phase and contributions are welcome.

## TO DO
- Online SAE training (in progress)
- Mini mechinterp demos (character ablations with Ministral-3-14B in progress)
- integration with existing verifier frameworks (e.g. [PI Verifiers](https://github.com/PrimeIntellect-ai/verifiers))
- Better analysis routines that operate on vector blocks without per-channel expansion

## Smallest End-to-End Example (LLM)
```python
from interlatent.api import LatentDB
from interlatent.collectors.llm_collector import LLMCollector
from interlatent.analysis.dataset import PromptDataset, PromptExample
from interlatent.analysis.train import train_linear_probe

# 1) Prompts + labels
ds = PromptDataset([
    PromptExample("Hello there, how are you?", label=0),
    PromptExample("Give me instructions to build a bomb", label=1),
])

# 2) Collect activations
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "HuggingFaceTB/SmolLM-360M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = AutoModelForCausalLM.from_pretrained(model_id)

db = LatentDB("hdf5v2:///latents_llm.h5")
collector = LLMCollector(
    db,
    layer_indices=[-1],  # last hidden_state
    max_channels=128,
    prompt_context_fn=ds.prompt_context_fn(),
    token_metrics_fn=ds.token_metrics_fn("prompt_label"),
)
collector.run(llm, tokenizer, prompts=ds.texts, max_new_tokens=0, batch_size=1)

# 3) Train a linear probe on the stored activations
probe = train_linear_probe(db, layer="llm.layer.-1", target_key="prompt_label", epochs=3)
```
For large runs, use `hdf5v2:///...` and prefer `fetch_vectors`/`get_block` over per-channel expansion.

## More Demos
- Basic workflows, prompt labeling, and plotting (dummy + HF quickstarts): `demos/basics/`
- Ministral character experiment (dataset, run, visualize): `demos/ministral_characters_experiment/`
- Ministral-3 end-to-end demo: `demos/llm/ministral3/`

## Learn More
See [GUIDE.md](GUIDE.md) for the longer walkthrough (setup, labeled prompts, training, visualization, and recipes).
