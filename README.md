# Interlatent

Interlatent is a lightweight interpretability toolkit where you can: save prompts and activations in an SQL database with context, attach labels, learn sparse latents (transcoders/SAEs) and probes, and quickly see which tokens or states drive them. The goal is to allow new independent researchers / engineers to dabble with understanding their models. It uses SQLite and small scripts by design, aimed at small/medium-scale experiments.

## Smallest End-to-End Example (LLM)
```python
from interlatent.api import LatentDB
from interlatent.collectors.llm_collector import LLMCollector
from interlatent.llm.prompt_dataset import PromptDataset, PromptExample
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

db = LatentDB("sqlite:///latents_llm.db")
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

## More Demos
- Ministral character experiment (dataset, run, visualize): `scripts/demos/ministral/character_ablations/`
- LLM workflow demo (no downloads): `scripts/demos/ministral/llm_workflow_demo.py`
- Real HF model demo: `scripts/demos/ministral/llm_real_model_demo.py` (set `RUN_LLM_REAL=1` and `LLM_MODEL`)
- Prompt labeling demo: `scripts/demos/ministral/prompt_dataset_demo.py`

## Learn More
See [GUIDE.md](GUIDE.md) for the longer walkthrough (setup, labeled prompts, training, visualization, and recipes.
