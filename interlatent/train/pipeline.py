from interlatent.train.trainer import TranscoderTrainer
from interlatent.train.dataset import ActivationPairDataset
from interlatent.schema import ActivationEvent

import torch
from torch.utils.data import DataLoader

class TranscoderPipeline:
    """
    Learn a sparse bottleneck for ONE layer.
      • Fetches activations logged as  {layer}:pre  and  {layer}:post
      • Writes latents back to DB as   latent:{layer}
    """

    def __init__(self, db, layer: str, *, k: int = 32, epochs: int = 5):
        self.db, self.layer, self.k, self.epochs = db, layer, k, epochs

    def run(self):
        ds = ActivationPairDataset(self.db, self.layer)
        loader = DataLoader(ds, batch_size=256, shuffle=True)

        trainer = TranscoderTrainer(ds.in_dim, ds.out_dim, self.k)
        trainer.train(loader, epochs=self.epochs)
        self._write_latents(trainer.T, ds)

        return trainer

    def _write_latents(self, encoder, dataset):
        latent_layer = f"latent:{self.layer}"
        print("latent_layer:", latent_layer)
        encoder.eval()

        with torch.no_grad():
            for step, (x_pre, _) in enumerate(dataset):
                z = encoder(x_pre.unsqueeze(0))          # (1, k)
                for idx, val in enumerate(z.squeeze(0)): # scalar per latent
                    self.db.write_event(
                        ActivationEvent(
                            run_id="latent_run",
                            step=step,
                            layer=latent_layer,
                            channel=idx,
                            tensor=[float(val)],
                            context={},
                            value_sum=float(val),
                            value_sq_sum=float(val * val),
                        )
                    )
        self.db.flush()
