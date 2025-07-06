import gymnasium as gym

# Stable-Baselines3 for the pretrained CartPole agent
from stable_baselines3 import PPO       

# Your LatentDB stack
from interlatent.metrics import LambdaMetric
from interlatent.collector import Collector
from interlatent.api import LatentDB
from interlatent.train.pipeline import TranscoderPipeline

# ------------------------------------------------------------------
# helper – you can replace with a local .zip path or a Zoo download
# ------------------------------------------------------------------
def load_solved_ppo_policy():
    """
    Returns a PPO policy that scores ~475 on CartPole.
    Replace the `PPO.load(...)` path with wherever you stash the checkpoint.
    """
    # If you have the SB3 Zoo checkpoint:
    #   wget https://huggingface.co/sb3/ppo-CartPole-v1/resolve/main/ppo-CartPole-v1.zip
    #   then point to that file:
    return PPO.load("pretrained/ppo-CartPole-v1.zip").policy


def test_latent_correlations():
    env = gym.make("CartPole-v1")
    model = load_solved_ppo_policy()         # stub: load SB3 checkpoint

    db = LatentDB("sqlite:///runs/latent_demo.db")
    collector = Collector(
        db,
        hook_layers=["mlp_extractor.policy_net.0", "mlp_extractor.policy_net.2"],
        metric_fns=[LambdaMetric("pole_angle", lambda obs, **_: obs[2])],
    )
    collector.run(model, env, steps=100)

    TranscoderPipeline(db, "mlp_extractor.policy_net.0", "mlp_extractor.policy_net.2", k=4, epochs=3).run()
    db.compute_stats(min_count=100)

    latents = list(db.iter_statblocks(layer="latent:policy_net.2"))
    assert any(abs(rho) > 0.7 for sb in latents for _, rho in sb.top_correlations)

