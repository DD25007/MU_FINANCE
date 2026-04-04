"""
Method 3: Gradient Ascent (GA).

Paper: Graves et al., "Amnesiac Machine Learning" (AAAI 2021); Yao et al. (2024)

The simplest unlearning method: ascend (maximize) the loss on the forget set to
make the model "unlearn" it. Fine-tunes ALL model weights (no adapters).

Known instability: retain AUC can collapse if run too long. Document this in results.
"""

import copy
import time
import torch
import torch.nn as nn
from itertools import cycle

from data.datasets import CreditDataset, make_loader
from train import evaluate


def unlearn(model, D_forget, D_retain, D_val, config, device):
    """
    Gradient ascent unlearning.

    config keys:
        ga_lr: float             (learning rate, default 1e-5)
        max_steps: int           (max gradient ascent steps, default 100)
        grad_clip: float         (gradient clipping norm, default 1.0)
        retain_auc_threshold: float (stop if retain AUC drops below, default 0.68)
        batch_size: int          (default 64)
    """
    model = copy.deepcopy(model).to(device)
    ga_lr = config.get("ga_lr", 1e-5)
    max_steps = config.get("max_steps", 100)
    grad_clip = config.get("grad_clip", 1.0)
    retain_auc_threshold = config.get("retain_auc_threshold", 0.68)
    batch_size = config.get("batch_size", 64)
    verbose = config.get("verbose", True)

    optimizer = torch.optim.Adam(model.parameters(), lr=ga_lr)
    criterion = nn.BCEWithLogitsLoss()

    forget_loader = make_loader(D_forget, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(D_val, batch_size=256, shuffle=False)
    forget_iter = cycle(iter(forget_loader))

    model.train()
    for step in range(1, max_steps + 1):
        x_num, x_cat, y = next(forget_iter)
        x_num = x_num.to(device) if x_num.numel() > 0 else None
        x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x_num, x_cat)
        # GRADIENT ASCENT: negate the loss to maximize it
        loss = -criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Early stopping: check retain-val AUC periodically
        if step % 10 == 0:
            retain_metrics = evaluate(model, val_loader, device)
            retain_auc = retain_metrics["auc"]

            if verbose and step % 50 == 0:
                print(f"    GA step {step}/{max_steps} | retain_auc={retain_auc:.4f}")

            if retain_auc < retain_auc_threshold:
                if verbose:
                    print(f"    GA: retain AUC collapsed to {retain_auc:.4f} at step {step} -- stopping")
                break

    return model


def gradient_ascent_unlearn(model, D_forget, D_retain, D_val, config, device):
    """Convenience wrapper. Returns (model, elapsed_seconds)."""
    start = time.perf_counter()
    result = unlearn(model, D_forget, D_retain, D_val, config, device)
    elapsed = time.perf_counter() - start
    return result, elapsed
