"""
Phase 2 — Forget Adapter (Gradient Ascent + GradDiff).

Trains a LoRA adapter Θ_f on a copy of the base model using:
  loss = -loss_forget + lambda_retain * loss_retain   (GradDiff)

The retain term keeps utility from collapsing so we can run more steps
aggressively. Best state is tracked as lowest forget_acc while retain_auc
stays above the minimum threshold.
"""

import copy
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Optional

from data.datasets import CreditDataset, make_loader
from models.lora import freeze_non_lora, count_parameters
from train import evaluate


def run_forget_adapter(
    model: nn.Module,
    forget_ds: CreditDataset,
    retain_ds: CreditDataset,
    device: torch.device,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    max_steps: int = 100,
    batch_size: int = 64,
    lr: float = 5e-4,
    retain_auc_min: float = 0.60,    # hard floor — stop if retain AUC drops below this
    lambda_retain: float = 0.5,      # GradDiff retain regularisation weight
    verbose: bool = True,
) -> tuple:
    """
    Attaches LoRA adapter to a COPY of model and trains via GradDiff on D_f.

    GradDiff loss = -BCE(forget) + lambda_retain * BCE(retain)
    This allows aggressive forgetting while constraining utility loss.

    Best checkpoint = lowest forget_acc with retain_auc >= retain_auc_min.

    Returns:
      model_with_forget_adapter  — model + Θ_f (retain adapter not yet applied)
      history                    — dict with metrics per step
    """
    model_fa = copy.deepcopy(model).to(device)

    lora_layers = model_fa.attach_lora(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    freeze_non_lora(model_fa)

    total_p, trainable_p = count_parameters(model_fa)
    if verbose:
        print(f"  [ForgetAdapter] LoRA params: {trainable_p:,} / {total_p:,} "
              f"({100*trainable_p/total_p:.2f}%)")

    forget_loader = make_loader(forget_ds, batch_size=batch_size, shuffle=True)
    retain_loader = make_loader(retain_ds, batch_size=batch_size, shuffle=True)
    retain_eval_loader = make_loader(retain_ds, batch_size=256, shuffle=False)

    optimizer = Adam(model_fa.get_lora_params(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {
        "step": [],
        "forget_loss_raw": [],
        "retain_auc": [],
        "forget_acc": [],
    }

    forget_iter = iter(forget_loader)
    retain_iter = iter(retain_loader)
    t0 = time.time()

    # Best state = lowest forget_auc while retain_auc is still healthy.
    # Using AUC (not accuracy) handles class-imbalanced datasets correctly —
    # on skewed data accuracy barely moves even when the model is forgetting.
    best_state = copy.deepcopy(model_fa.state_dict())
    best_forget_auc = float("inf")

    for step in range(1, max_steps + 1):
        model_fa.train()

        # --- forget batch ---
        try:
            batch_f = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            batch_f = next(forget_iter)

        xn_f, xc_f, y_f = batch_f
        xn_f = xn_f.to(device) if xn_f.numel() > 0 else None
        xc_f = xc_f.to(device) if xc_f.numel() > 0 else None
        y_f = y_f.to(device)

        # --- retain batch (for GradDiff regularisation) ---
        try:
            batch_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            batch_r = next(retain_iter)

        xn_r, xc_r, y_r = batch_r
        xn_r = xn_r.to(device) if xn_r.numel() > 0 else None
        xc_r = xc_r.to(device) if xc_r.numel() > 0 else None
        y_r = y_r.to(device)

        optimizer.zero_grad()

        # GradDiff: ascend on forget, descend on retain simultaneously
        loss_f = criterion(model_fa(xn_f, xc_f), y_f)
        loss_r = criterion(model_fa(xn_r, xc_r), y_r)
        loss = -loss_f + lambda_retain * loss_r

        loss.backward()
        nn.utils.clip_grad_norm_(model_fa.get_lora_params(), 1.0)
        optimizer.step()

        if step % 5 == 0 or step == 1:
            retain_metrics = evaluate(model_fa, retain_eval_loader, device)
            forget_metrics = evaluate(model_fa, make_loader(forget_ds, 256, False), device)

            history["step"].append(step)
            history["forget_loss_raw"].append(loss_f.item())
            history["retain_auc"].append(retain_metrics["auc"])
            history["forget_acc"].append(forget_metrics["acc"])

            if verbose:
                print(f"    step {step:3d} | forget_auc={forget_metrics['auc']:.4f} "
                      f"forget_acc={forget_metrics['acc']:.3f} "
                      f"retain_auc={retain_metrics['auc']:.4f}")

            # Save checkpoint with lowest forget_auc while retain is healthy.
            # This gives Phase 3 a good starting point (reasonable retain_auc
            # to recover from). If retain has already collapsed, Phase 3 cannot
            # repair it without restoring forget too.
            if (retain_metrics["auc"] >= retain_auc_min and
                    forget_metrics["auc"] < best_forget_auc):
                best_forget_auc = forget_metrics["auc"]
                best_state = copy.deepcopy(model_fa.state_dict())

            # Hard floor: stop if retain collapses completely
            if retain_metrics["auc"] < retain_auc_min:
                if verbose:
                    print(f"    [ForgetAdapter] Retain AUC below {retain_auc_min:.2f} — stopping early")
                break

    elapsed = time.time() - t0
    # Restore the checkpoint with best (lowest) forget_acc
    model_fa.load_state_dict(best_state)

    history["elapsed"] = elapsed
    if verbose:
        print(f"  [ForgetAdapter] Done in {elapsed:.1f}s")

    return model_fa, history
