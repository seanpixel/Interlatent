# Interlatent
An open-source tool for peeking into both RL agents and language models by collecting, storing, and visualizing internal activations.


## Background
Back in June 2025, I experimented in interpreting simple RL models adopting the [transcoder methodology](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) developed by Anthropic to understand LLMs. Largely inspired by language model interpretability research, my hypothesis was that we can also probe RL agents by tracking how their internal activations affect their behavior (feature extraction) and steer policy by manipulating inner workings (intervention). RL environments supply a large pool of signal where we can obtain metrics to correlate with potentially interpretable activations to gain insights such as "feature 4 of layer 2 spikes when the Pacman agent approaches a power pellet". I wrote a blog post summarizing [this proof of concept](https://seanpixel.com/interpretable-rl-via-multiscale-transcoders-sparse-bottlenecks-at-2-4-and-8-dimensions) and spent weeks applying it to more complex RL environments. This repository serves as a tool for replicating this research for any and all models and environments—and now includes language models for interpretability workflows without RL.


## How it works
Interlatent streamlines extracting and inspecting activations:
1. Collect activations of specific layers from RL policies or HuggingFace LMs over rollouts/prompts, storing them in a SQLite-backed LatentDB.
2. (Optional) Train transcoders that bottleneck linear layers to produce sparse latent features.
3. Track and record metrics (RL env rewards, custom signals) to correlate with activations.
4. Visualize and explore activations: per-token traces, cross-prompt latent views, and quick DB summaries.


## Current Progress
The framework can hook onto given layers to collect activations, train transcoders using those activations, and compute correlations between each feature and metrics. It now includes HuggingFace LLM collectors and visualization utilities (per-token plots and cross-prompt latent plots). Scaling up to bigger models/environments will require more compute; this repo is a working but scrappy toolkit.

## How to Use
- RL activations: see `tests/test_end_to_end.py`.
- LLM activations: run `python tests/llm_1.py` (env `LLM_MODEL` overrides the default HF id) to populate a SQLite DB.
- Visuals: `python -m interlatent.vis.summary latents_llm.db` for table-style summaries; `python -m interlatent.vis.plot latents_llm.db --layer llm.layer.-1 --channel 0 --prompt-index 0` for per-prompt traces; `--all-prompts` plots a latent across prompts. `tests/vis_demo.py` shows end-to-end collection + plot generation.
- User-facing documentation is WIP; APIs may change.


## Future Plans & Motivations
Unlike current reasoning language models, RL-trained real-world agents carry out actions without “reasoning traces", which means that we have no idea why the model does what it does and what thoughts it has in mind while performing an action (technically, reasoning traces are black-boxes too but RL is worse). 

While carrying out previous experiments interpreting the cartpole agent with transcoders, I noticed myself asking LLMs to decipher latent activations by feeding it information such as correlation with acceleration, position, pole angular velocity etc. Now I ask myself: What if we do this real-time on a more complex environment? What if we can get an explanation of a humanoid agent’s mind asynchronously while they act? 

This tool serves as an initial stepping-stone to creating human-in-loop AI systems where we will eventually be able to understand real-time: why manufacturing robots make mistakes in edge-cases, why Tesla humanoids will take certain actions, and what biases Waymos might have.
