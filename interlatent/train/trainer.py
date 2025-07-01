import torch.nn as nn
import torch.nn.functional as F

def train_linear_transcoder(
        transcoder,
        pre_acts, post_acts,          # lists of Tensors
        *, epochs=10, lr=1e-3, l1=1e-3,
        device="cuda", max_samples=None):

    assert len(pre_acts) == len(post_acts)
    opt = torch.optim.Adam(transcoder.parameters(), lr=lr)

    for ep in range(epochs):
        total, n = 0., 0
        # fresh permutation every epoch (good for SGD)
        order = torch.randperm(len(pre_acts))
        if max_samples:
            order = order[:max_samples]

        for idx in order:
            x_pre  = pre_acts[idx].to(device).view(-1, 3136).float()  # (B,3136)
            y_true = post_acts[idx].to(device).view(-1, 512).float()  # (B,512)

            z, y_pred = transcoder(x_pre)
            loss = F.mse_loss(y_pred, y_true) + l1 * z.abs().mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item(); n += 1
        print(f"epoch {ep+1}/{epochs} | loss={total/n:.4f}")

    return transcoder