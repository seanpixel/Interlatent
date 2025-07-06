from interlatent.train.trainer import TranscoderTrainer
from interlatent.train.dataset import ActivationPairDataset
from interlatent.schema import ActivationEvent

import torch
from torch.utils.data import DataLoader

class TranscoderPipeline:
    def __init__(self, db, layer_pre, layer_post, *, k=32, epochs=5):
        self.db = db; self.layer_pre = layer_pre; self.layer_post = layer_post
        self.k = k; self.epochs = epochs

    def run(self):
        ds = ActivationPairDataset(self.db, self.layer_pre, self.layer_post)
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        trainer = TranscoderTrainer(ds.in_dim, ds.out_dim, self.k)
        ckpt = trainer.train(loader, epochs=self.epochs)
        self._write_latents(trainer.T, loader.dataset)

    def _write_latents(self, encoder, dataset):
        latent_layer = f"latent:{self.layer_post}"
        encoder.eval()
        with torch.no_grad():
            for step, (x_pre, _) in enumerate(dataset):
                z = encoder(x_pre.unsqueeze(0))  # shape (1, k)
                for idx, val in enumerate(z.squeeze(0)):
                    ev = ActivationEvent(
                        run_id="latent_run",
                        step=step,
                        layer=latent_layer,
                        channel=idx,
                        tensor=[float(val)],
                        context={},
                        value_sum=float(val),
                        value_sq_sum=float(val * val),
                    )
                    self.db.write_event(ev)
        self.db.flush()
