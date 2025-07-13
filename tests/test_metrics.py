from interlatent.metrics import LambdaMetric, EpisodeAccumulator
from interlatent.collector import Collector
from interlatent.api import LatentDB
import gymnasium as gym, torch.nn as nn

angle = LambdaMetric("pole_angle", lambda obs, **_: obs[2])
reward = EpisodeAccumulator("episode_reward", lambda *, reward, **__: reward)

db = LatentDB("sqlite:///runs/demo.db")
collector = Collector(db, metric_fns=[angle, reward])
env = gym.make("CartPole-v1")
model = nn.Sequential(nn.Linear(4, 8))

collector.run(model, env, steps=100)
db.flush()         # until we move flush into Collector internally
