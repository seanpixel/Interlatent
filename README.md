# Interlatent
an open-source tool for peeking into RL models using transcoder activations


## Background
Back in June 2025, I experimented in interpreting simple RL models adopting the [transcoder methodology](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) developed by Anthropic to understand LLMs. Largely inspired by language model interpretability research, my hypothesis was that we can also probe RL agents by tracking how their internal activations affect their behavior (feature extraction) and steer its policy by manipulating its inner workings (intervention). RL environments supply a large pool of signal where we can obtain metrics to correlate with potentially interpretable activations to gain insights such as "feature 4 of layer 2 spike when the Pacman agent approaches a power pellet". I wrote a blog post summarizing [this proof of concept](https://seanpixel.com/interpretable-rl-via-multiscale-transcoders-sparse-bottlenecks-at-2-4-and-8-dimensions) and spent weeks applying it to other more complex RL environments. This repository serves as a tool for replicating this research for any and all models and environments.


## How it works
Interlatent streamlines this process of extracting interpretable features:
1. Allow users to collect activations of specific layers of a model over several rollouts.
2. Use the activations to train transcoders that bottleneck linear layers of an AI model to produce sparse (not interpretable yet!) features.  
3. Track and record specific metrics from the RL environment.
4. Find patterns between the metrics and feature activations of the transcoder to make sparse features interpretable.  


## Current Progress
The framework can hook onto given layers of a model to collect activations, train transcoders using those activations, and compute correlations between each feature of the transcoder and any given RL environment metric. Scaling this up in order to test this on bigger models and complex environments require compute. Right now, the repository is a working, but scrappy, framework.


## Future Plans & Motivations
Unlike current reasoning language models, RL-trained real-world agents carry out actions without “<reasoning-trace>”s, which means that we have no idea why the model does what it does and what thoughts it has in mind while performing an action (technically, reasoning traces are black-boxes too but RL is worse). 

While carrying out previous experiments interpreting the cartpole agent with transcoders, I noticed myself asking LLMs to decipher latent activations by feeding it information such as correlation with acceleration, position, pole angular velocity etc. Now I ask myself: What if we do this real-time on a more complex environment? What if we can get an explanation of a humanoid agent’s mind asynchronously while they act? 

This tool serves as an initial stepping-stone to creating human-in-loop AI systems where we will eventually be able to understand real-time: why manufacturing robots make mistakes in edge-cases, why Tesla humanoids will take certain actions, and what biases Waymos might have.
