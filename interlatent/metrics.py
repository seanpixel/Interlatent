from __future__ import annotations
from typing import Protocol, Any, Dict, Callable, Optional


class Metric(Protocol):
    """
    A “metric” produces one scalar per timestep and can reset at episode-end.
    """

    name: str

    def reset(self) -> None: ...
    def step(self, *, obs, reward, info) -> Optional[float]: ...


class LambdaMetric:
    """
    Wrap any `(obs, reward, info) -> scalar` into a Metric.
    Example
    -------
    pole_ang = LambdaMetric("pole_angle", lambda obs, **_: float(obs[2]))
    """

    def __init__(self, name: str, fn: Callable[..., float | None]):
        self.name, self._fn = name, fn

    def reset(self) -> None:
        # stateless lambda never needs resetting
        pass

    def step(self, *, obs, reward, info):
        return self._fn(obs=obs, reward=reward, info=info)


class EpisodeAccumulator:
    """
    Accumulates a per-step value (e.g., reward) over an episode,
    emits the running total every step, and resets at `env.reset()`.
    """

    def __init__(self, name: str, fn: Callable[..., float]):
        self.name, self._fn = name, fn
        self._acc = 0.0

    def reset(self) -> None:
        self._acc = 0.0

    def step(self, *, obs, reward, info):
        self._acc += self._fn(obs=obs, reward=reward, info=info)
        return self._acc

__all__ = ["Metric", "LambdaMetric", "EpisodeAccumulator"]
