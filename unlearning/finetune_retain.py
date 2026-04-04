"""
Method 6: Fine-tune on Retain Set Only.

The simplest baseline: after training on full D, just continue training on
D_retain only. This is the "catastrophic forgetting" baseline.

Expected result: MIA score will NOT drop to 50% -- this method doesn't truly
unlearn. Include it because reviewers expect it and it provides a lower bound.
"""

import copy
import time
import torch
import torch.nn as nn

from data.datasets import CreditDataset, make_loader
from train import train_model


def unlearn(model, D_forget, D_retain, D_val, config, device):
    """
    Fine-tune the already-trained model on D_retain only.
    Simple, fast, but usually doesn't achieve true forgetting.

    config keys:
        lr: float (default 1e-5, smaller than original training)
        epochs: int (default 5-10)
        batch_size: int (default 256)
        patience: int (default 5)
    """
    model = copy.deepcopy(model).to(device)
    lr = config.get("lr", 1e-5)
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 256)
    patience = config.get("patience", 5)
    verbose = config.get("verbose", True)

    model, history = train_model(
        model, D_retain, D_val, device,
        epochs=epochs, batch_size=batch_size,
        lr=lr, patience=patience,
        verbose=verbose,
    )

    return model


def finetune_retain_unlearn(model, D_retain, D_val, config, device):
    """Convenience wrapper. Returns (model, elapsed_seconds)."""
    start = time.perf_counter()
    result = unlearn(model, None, D_retain, D_val, config, device)
    elapsed = time.perf_counter() - start
    return result, elapsed
