from __future__ import annotations

from typing import Literal

import torch
from torch.utils.data import DataLoader

from interlatent.analysis.dataset import LinearProbeDataset
from interlatent.analysis.models.linear_probe import LinearProbe


def train_linear_probe(
    db,
    layer: str,
    target_key: str,
    *,
    task: Literal["regression", "classification"] = "regression",
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str | torch.device | None = None,
    limit: int | None = None,
):
    """
    Train a simple linear probe on activations of *layer* to predict *target_key*
    found in ActivationEvent.context["metrics"] (or directly in context).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = LinearProbeDataset(db, layer=layer, target_key=target_key, limit=limit)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    out_dim = 1
    if task == "classification":
        # Infer num classes from dataset labels
        labels = torch.stack([y for _, y in ds.samples])
        classes = torch.unique(labels.long())
        out_dim = int(classes.max().item()) + 1

    model = LinearProbe(ds.in_dim, out_dim=out_dim)
    model.to(device)

    if task == "classification":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if task == "classification":
                logits = model(x)
                loss = criterion(logits, y.long())
            else:
                pred = model(x).squeeze(-1)
                loss = criterion(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model.cpu()
