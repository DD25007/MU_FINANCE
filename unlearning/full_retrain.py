"""
Method 1: Full Retrain (Gold Standard).

Retrain the SAME model architecture from scratch on D_retain only (D_train minus D_forget).
This is the oracle baseline -- what would a model look like that never saw D_forget?

This is SLOW but gives the gold standard for comparison.
"""

import copy
import time
import torch
import torch.nn as nn

from data.datasets import CreditDataset, make_loader
from train import train_model


def unlearn(model, D_forget, D_retain, D_val, config, device):
    """
    Full retrain unlearning -- does NOT use the pretrained model.
    Trains a fresh model from scratch on D_retain only.

    config keys:
        model_factory: callable() -> fresh nn.Module
        lr: float (default 1e-4)
        wd: float (default 1e-5)
        max_epochs: int (default 50)
        patience: int (default 10)
        batch_size: int (default 256)
    """
    model_factory = config.get("model_factory")
    if model_factory is None:
        raise ValueError("full_retrain requires 'model_factory' in config")

    fresh_model = model_factory().to(device)
    lr = config.get("lr", 1e-4)
    wd = config.get("wd", 1e-5)
    max_epochs = config.get("max_epochs", 50)
    patience = config.get("patience", 10)
    batch_size = config.get("batch_size", 256)

    fresh_model, history = train_model(
        fresh_model, D_retain, D_val, device,
        epochs=max_epochs, batch_size=batch_size,
        lr=lr, weight_decay=wd, patience=patience,
        verbose=config.get("verbose", True),
    )

    return fresh_model


def full_retrain(model_factory, D_retain, D_val, config, device):
    """
    Convenience wrapper matching the spec's function signature.
    Returns (model, elapsed_seconds).
    """
    config = {**config, "model_factory": model_factory}
    start = time.perf_counter()
    model = unlearn(None, None, D_retain, D_val, config, device)
    elapsed = time.perf_counter() - start
    return model, elapsed
