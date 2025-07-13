import gymnasium as gym
from stable_baselines3 import PPO

from interlatent.metrics import LambdaMetric
from interlatent.collector import Collector
from interlatent.api import LatentDB
from interlatent.train.pipeline import TranscoderPipeline

# --------------------------------------------------------
def load_solved_ppo_policy():
    # path can be swapped for your own checkpoint
    return PPO.load("pretrained/ppo-CartPole-v1.zip").policy


def test_transcoder_saved_and_correlated(tmp_path):
    db = LatentDB(f"sqlite:///{tmp_path}/latents.db")

    env   = gym.make("CartPole-v1")
    model = load_solved_ppo_policy()

    collector = Collector(
        db,
        hook_layers=["mlp_extractor.policy_net.0"],
        metric_fns=[LambdaMetric("pole_angle", lambda obs, **_: float(obs[2]))],
    )
    collector.run(model, env, steps=200)          # small run for CI

    pipe = TranscoderPipeline(db, "mlp_extractor.policy_net.0", k=4, epochs=3)
    trainer = pipe.run()                          # trains, saves, back-fills

    # artifact row exists
    artifacts = list(db._store._conn.execute("SELECT kind, path FROM artifacts"))
    assert any(a["kind"] == "transcoder" for a in artifacts)

    # compute stats to get correlations
    db.compute_stats(min_count=1)

    strong = []
    for sb in db.iter_statblocks(layer="latent:mlp_extractor.policy_net.0"):
        print(sb.top_correlations)
        strong.extend(abs(rho) for _, rho in sb.top_correlations if abs(rho) > 0.5)

    assert strong, "Expected at least one latent with |ρ| > 0.5 against pole_angle"

    db.close()
