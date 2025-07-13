from interlatent.metrics import LambdaMetric
from interlatent.collector import Collector
from interlatent.api import LatentDB
import torch.nn as nn, gymnasium as gym

# 1. policy
model = nn.Sequential(nn.Linear(4, 8))        # layer name will be '0'

# 2. metric(s) you care about
angle = LambdaMetric("pole_angle", lambda obs, **_: float(obs[2]))

# 3. LatentDB + Collector (register the layer!)
db = LatentDB("sqlite:///runs/cartpole_raw.db")
collector = Collector(
    db,
    hook_layers=["0"],          # <-- this is the critical bit
    metric_fns=[angle],
)

# 4. run env
env = gym.make("CartPole-v1")
collector.run(model, env, steps=100)
db.flush()               # persist events
db.compute_stats(min_count=1)       # moments + correlations

# 5. inspect
for sb in db.iter_statblocks(layer="0"):
    print(f"ch {sb.channel} correlations → {sb.top_correlations}")
