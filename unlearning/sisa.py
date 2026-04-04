"""
Method 2: SISA Training (Sharded, Isolated, Sliced, Aggregated).

Paper: Bourtoule et al., "Machine Unlearning" (IEEE S&P 2021)

SISA divides training data into S shards. At unlearn time, only the shard(s)
containing the forget point(s) are retrained. Predictions are aggregated by
averaging logits across shard models.

Hyperparams to sweep: n_shards in {2, 5, 10}
Key metric: fraction of data retrained vs full retrain (efficiency gain).
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

from data.datasets import CreditDataset, make_loader, subset_dataset
from train import train_model


class SISAEnsemble(nn.Module):
    """Wraps a list of shard models. Forward averages their logits."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x_num, x_cat=None):
        if len(self.models) == 0:
            raise RuntimeError("SISA ensemble has no shard models after unlearning.")
        logits = torch.stack([m(x_num, x_cat) for m in self.models], dim=0)
        return logits.mean(dim=0)


def sisa_train(model_factory, full_train_ds, val_ds, n_shards, config, device):
    """
    Phase 1 (done before unlearning request):
    Train S independent models, one per shard.
    Returns: (shard_models, shards) where shards is list of index arrays.
    """
    n_total = len(full_train_ds)
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    shard_size = n_total // n_shards
    shards = [indices[i * shard_size:(i + 1) * shard_size] for i in range(n_shards)]
    # Assign remainder to last shard
    if n_total % n_shards != 0:
        shards[-1] = np.append(shards[-1], indices[n_shards * shard_size:])

    epochs = config.get("epochs_per_shard", 15)
    batch_size = config.get("batch_size", 256)
    lr = config.get("lr", 1e-3)
    verbose = config.get("verbose", True)

    shard_models = []
    for i, shard_idx in enumerate(shards):
        if verbose:
            print(f"    Training shard {i+1}/{n_shards} ({len(shard_idx)} samples)")
        shard_ds = subset_dataset(full_train_ds, shard_idx)
        model = model_factory().to(device)
        model, _ = train_model(
            model, shard_ds, val_ds, device,
            epochs=epochs, batch_size=batch_size,
            lr=lr, verbose=False, patience=5,
        )
        shard_models.append(model)

    return shard_models, shards


def sisa_unlearn_shards(shard_models, shards, forget_indices, full_train_ds,
                        val_ds, model_factory, config, device):
    """
    Phase 2 (at unlearn time):
    Find which shard(s) contain forget_indices.
    Retrain only those shards from scratch without the forget samples.
    """
    forget_set = set(forget_indices.tolist()) if hasattr(forget_indices, 'tolist') else set(forget_indices)
    verbose = config.get("verbose", True)
    epochs = config.get("epochs_per_shard", 15)
    batch_size = config.get("batch_size", 256)
    lr = config.get("lr", 1e-3)

    # Identify affected shards
    affected = []
    for i, shard_idx in enumerate(shards):
        if any(int(idx) in forget_set for idx in shard_idx):
            affected.append(i)

    if verbose:
        print(f"    Affected shards: {affected} (out of {len(shards)})")
        frac_retrained = sum(len(shards[i]) for i in affected) / sum(len(s) for s in shards)
        print(f"    Fraction of data retrained: {frac_retrained:.1%}")

    for shard_i in affected:
        # Remove forget points from this shard
        clean_idx = np.array([idx for idx in shards[shard_i] if int(idx) not in forget_set])
        if len(clean_idx) == 0:
            if verbose:
                print(f"    Shard {shard_i} is empty after removing forget samples -- dropping shard from ensemble")
            shard_models[shard_i] = None
            continue

        clean_ds = subset_dataset(full_train_ds, clean_idx)
        new_model = model_factory().to(device)
        new_model, _ = train_model(
            new_model, clean_ds, val_ds, device,
            epochs=epochs, batch_size=batch_size,
            lr=lr, verbose=False, patience=5,
        )
        shard_models[shard_i] = new_model

    shard_models = [model for model in shard_models if model is not None]
    if len(shard_models) == 0:
        raise RuntimeError("All SISA shards became empty after removing forget samples.")
    return shard_models


def unlearn(model, D_forget, D_retain, D_val, config, device):
    """
    Unified unlearn interface for SISA.

    config keys:
        model_factory: callable() -> fresh nn.Module
        full_train_ds: CreditDataset (full training set)
        forget_indices: np.ndarray (indices into full_train_ds)
        n_shards: int (default 10)
        epochs_per_shard: int (default 15)
        batch_size: int (default 256)
        lr: float (default 1e-3)
    """
    model_factory = config["model_factory"]
    full_train_ds = config["full_train_ds"]
    forget_indices = config["forget_indices"]
    n_shards = config.get("n_shards", 10)
    verbose = config.get("verbose", True)

    if verbose:
        print(f"  [SISA] Training {n_shards} shards...")

    # Phase 1: Train all shards
    shard_models, shards = sisa_train(
        model_factory, full_train_ds, D_val, n_shards, config, device
    )

    # Phase 2: Retrain affected shards
    if verbose:
        print(f"  [SISA] Unlearning from affected shards...")
    shard_models = sisa_unlearn_shards(
        shard_models, shards, forget_indices, full_train_ds,
        D_val, model_factory, config, device
    )

    # Return ensemble
    ensemble = SISAEnsemble(shard_models)
    return ensemble


def sisa_full(model_factory, full_train_ds, forget_indices, val_ds, config, device):
    """Convenience wrapper. Returns (ensemble_model, elapsed_seconds)."""
    config = {
        **config,
        "model_factory": model_factory,
        "full_train_ds": full_train_ds,
        "forget_indices": forget_indices,
    }
    start = time.perf_counter()
    ensemble = unlearn(None, None, None, val_ds, config, device)
    elapsed = time.perf_counter() - start
    return ensemble, elapsed
