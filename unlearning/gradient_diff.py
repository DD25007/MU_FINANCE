"""
Method 4: Gradient Difference (GradDiff).

Paper: Liu et al., "Continual Learning and Private Unlearning" (2022)

Improvement over pure GA: simultaneously ascend on forget set AND descend on
retain set. The retain gradient stabilizes training and prevents catastrophic
forgetting of retain performance.

This is typically the STRONGEST baseline -- expect it to perform best among
non-LoRA methods.

Hyperparams to sweep: alpha in {0.1, 0.5, 1.0, 2.0}, lr in {1e-6, 1e-5, 5e-5}
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
    Gradient difference unlearning.

    config keys:
        lr: float                (learning rate, default 1e-5)
        alpha: float             (weight on retain loss, default 1.0)
        max_steps: int           (default 200)
        grad_clip: float         (gradient clipping norm, default 1.0)
        retain_auc_threshold: float (stop if retain AUC drops below, default 0.70)
        batch_size: int          (default 64)
    """
    model = copy.deepcopy(model).to(device)
    lr = config.get("lr", 1e-5)
    alpha = config.get("alpha", 1.0)
    max_steps = config.get("max_steps", 200)
    grad_clip = config.get("grad_clip", 1.0)
    retain_auc_threshold = config.get("retain_auc_threshold", 0.70)
    batch_size = config.get("batch_size", 64)
    verbose = config.get("verbose", True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    forget_loader = make_loader(D_forget, batch_size=batch_size, shuffle=True)
    retain_loader = make_loader(D_retain, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(D_val, batch_size=256, shuffle=False)

    forget_iter = cycle(iter(forget_loader))
    retain_iter = cycle(iter(retain_loader))

    model.train()
    for step in range(1, max_steps + 1):
        # Forget batch -- ASCEND (maximize loss)
        xf_num, xf_cat, yf = next(forget_iter)
        xf_num = xf_num.to(device) if xf_num.numel() > 0 else None
        xf_cat = xf_cat.to(device) if xf_cat.numel() > 0 else None
        yf = yf.to(device)

        forget_logits = model(xf_num, xf_cat)
        loss_forget = -criterion(forget_logits, yf)  # negate = ascend

        # Retain batch -- DESCEND (minimize loss, normal training)
        xr_num, xr_cat, yr = next(retain_iter)
        xr_num = xr_num.to(device) if xr_num.numel() > 0 else None
        xr_cat = xr_cat.to(device) if xr_cat.numel() > 0 else None
        yr = yr.to(device)

        retain_logits = model(xr_num, xr_cat)
        loss_retain = criterion(retain_logits, yr)

        # Combined loss: ascend on forget, descend on retain
        total_loss = loss_forget + alpha * loss_retain

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Early stopping on val AUC
        if step % 20 == 0:
            val_metrics = evaluate(model, val_loader, device)
            val_auc = val_metrics["auc"]

            if verbose and step % 50 == 0:
                print(f"    GradDiff step {step}/{max_steps} | val_auc={val_auc:.4f} "
                      f"| loss_f={loss_forget.item():.4f} loss_r={loss_retain.item():.4f}")

            if val_auc < retain_auc_threshold:
                if verbose:
                    print(f"    GradDiff: val AUC dropped to {val_auc:.4f} at step {step} -- stopping")
                break

    return model


def gradient_diff_unlearn(model, D_forget, D_retain, D_val, config, device):
    """Convenience wrapper. Returns (model, elapsed_seconds)."""
    start = time.perf_counter()
    result = unlearn(model, D_forget, D_retain, D_val, config, device)
    elapsed = time.perf_counter() - start
    return result, elapsed
