import torch as th
from stable_baselines3.common.callbacks import BaseCallback

class ActivationCatcher(BaseCallback):
    """
    Records pre- and post-activations for:

        features_extractor.cnn.4   →  labelled  conv4_pre / conv4_post
        features_extractor.linear.0 → labelled  linear0_pre / linear0_post

    Nothing else is touched.
    """

    def __init__(self,
                 sample_every: int = 20,
                 taps: tuple[str, ...] = ("features_extractor.cnn.4",
                                           "features_extractor.linear.0"),
                 verbose: int = 0):
        super().__init__(verbose)
        self.sample_every = sample_every
        self.taps         = set(taps)
        self.buffer       = {f"{self._short(n)}_{side}": []
                             for n in self.taps
                             for side in ("pre", "post")}
        self.meta   = []
        self.hooks  = []
        self.verbose = verbose

    # ---------- helpers ----------
    @staticmethod
    def _short(name: str) -> str:
        return "conv4" if "cnn.4" in name else "linear0"

    def _make_pre(self, key):
        def pre_hook(_, inputs):
            # inputs is a tuple; we want the first element
            x = inputs[0].detach().cpu()
            self.buffer[key].append(x)
        return pre_hook

    def _make_post(self, key):
        def post_hook(_, __, output):
            y = output.detach().cpu()
            self.buffer[key].append(y)
        return post_hook
    # --------------------------------

    def _on_training_start(self):
        for name, module in self.model.policy.named_modules():
            if name in self.taps:
                short = self._short(name)
                self.hooks.append(
                    module.register_forward_pre_hook(
                        self._make_pre(f"{short}_pre")
                    )
                )
                self.hooks.append(
                    module.register_forward_hook(
                        self._make_post(f"{short}_post")
                    )
                )
                if self.verbose:
                    print(f"Hooked {name}  →  {short}_pre / {short}_post")

    # rollout bookkeeping (optional)
    def _on_rollout_end(self):
        if (self.num_timesteps // self.model.n_steps) % self.sample_every == 0:
            rb = self.model.rollout_buffer
            self.meta.append(dict(
                rewards=rb.rewards.copy(),
                actions=rb.actions.copy()
            ))

    def _on_step(self):           # keep the training loop alive
        return True

    def _on_training_end(self):   # tidy up
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
