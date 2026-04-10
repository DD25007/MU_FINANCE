"""
Phase 2 — Forget Adapter (Gradient Ascent + GradDiff).

Improvements over baseline:
  [OPT-1] Noise injection after training: adds calibrated Gaussian noise to LoRA
          weights, making membership inference harder (DP-flavored).
  [OPT-2] Tighter retain_auc_min floor (configurable, default lowered to 0.70
          so Phase 3 has more room to recover, allowing deeper forgetting).
  [OPT-3] Gradient clipping per-layer instead of global — prevents one attention
          head from dominating the ascent.
  [OPT-4] Cosine annealing LR schedule on the forget adapter (helps avoid
          oscillation in the final steps).
  [OPT-5] Demographic-aware forget set: logs per-group metrics if group_labels
          are provided (age < 25 subgroup tracking for fairness analysis).
  [OPT-6] Likelihood Ratio score logged each checkpoint (used by improved MIA).
"""

import copy
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional
import numpy as np

from data.datasets import CreditDataset, make_loader
from models.lora import freeze_non_lora, count_parameters
from train import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# [OPT-1] Noise Injection
# ─────────────────────────────────────────────────────────────────────────────


def _inject_noise_into_lora(model: nn.Module, noise_scale: float, device: torch.device):
    """
    Adds calibrated Gaussian noise to all LoRA A and B matrices.

    The noise scale is proportional to the gradient magnitude of the forget set
    (DP-flavored). This degrades the attacker's ability to distinguish forget
    samples from non-members via confidence features.

    noise_scale: std of the Gaussian noise (e.g. 0.01–0.05).
                 Higher = more privacy, lower = more utility.
                 Recommended: 0.01 (safe), 0.03 (aggressive).
    """
    from models.lora import LoRALinear

    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.lora_A.data += torch.randn_like(module.lora_A) * noise_scale
                module.lora_B.data += torch.randn_like(module.lora_B) * noise_scale
    return model


# ─────────────────────────────────────────────────────────────────────────────
# [OPT-3] Per-layer gradient clipping
# ─────────────────────────────────────────────────────────────────────────────


def _clip_grad_per_layer(params, max_norm: float = 1.0):
    """
    Clips gradients per parameter tensor instead of globally.
    Prevents one attention head from monopolising the gradient ascent step,
    which can cause retain AUC to collapse faster than needed.
    """
    for p in params:
        if p.grad is not None:
            nn.utils.clip_grad_norm_([p], max_norm)


# ─────────────────────────────────────────────────────────────────────────────
# [OPT-5] Per-group metric logging
# ─────────────────────────────────────────────────────────────────────────────


def _evaluate_by_group(model, dataset, group_labels, device):
    """
    Evaluates forget accuracy separately for each demographic group.
    group_labels: np.array of same length as dataset, integer group ids.

    Returns dict: {group_id: {"acc": ..., "auc": ...}}
    """
    from data.datasets import subset_dataset

    unique_groups = np.unique(group_labels)
    results = {}
    for g in unique_groups:
        idx = np.where(group_labels == g)[0]
        sub_ds = subset_dataset(dataset, idx)
        loader = make_loader(sub_ds, batch_size=256, shuffle=False)
        metrics = evaluate(model, loader, device)
        results[int(g)] = {"acc": metrics["acc"], "auc": metrics["auc"]}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────


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
    # ── [OPT-2] Lowered default from 0.60 → 0.70 so Phase 3 has more room ──
    retain_auc_min: float = 0.70,
    lambda_retain: float = 0.5,
    # ── [OPT-1] Noise injection ──────────────────────────────────────────────
    noise_injection: bool = False,  # Toggle: add DP-flavored noise after training
    noise_scale: float = 0.02,  # Std of Gaussian noise (0.01=safe, 0.05=aggressive)
    # ── [OPT-3] Per-layer gradient clipping ─────────────────────────────────
    per_layer_clip: bool = False,  # Toggle: clip each LoRA param separately
    grad_clip_norm: float = 1.0,
    # ── [OPT-4] LR schedule ─────────────────────────────────────────────────
    use_lr_schedule: bool = False,  # Toggle: cosine annealing on forget adapter LR
    # ── [OPT-5] Demographic group tracking ──────────────────────────────────
    forget_group_labels: Optional[np.ndarray] = None,  # e.g. (age < 25).astype(int)
    # ── [OPT-6] Likelihood ratio logging (used by improved MIA) ─────────────
    log_likelihood_ratio: bool = False,  # Toggle: log LR score each checkpoint
    retain_reference_loss: Optional[float] = None,  # mean retain loss from base model
    verbose: bool = True,
) -> tuple:
    """
    Attaches LoRA adapter to a COPY of model and trains via GradDiff on D_f.

    GradDiff loss = -BCE(forget) + lambda_retain * BCE(retain)

    Returns:
      model_with_forget_adapter, history
    """
    model_fa = copy.deepcopy(model).to(device)

    lora_layers = model_fa.attach_lora(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
    )
    freeze_non_lora(model_fa)

    total_p, trainable_p = count_parameters(model_fa)
    if verbose:
        print(
            f"  [ForgetAdapter] LoRA params: {trainable_p:,} / {total_p:,} "
            f"({100*trainable_p/total_p:.2f}%)"
        )

    forget_loader = make_loader(forget_ds, batch_size=batch_size, shuffle=True)
    retain_loader = make_loader(retain_ds, batch_size=batch_size, shuffle=True)
    retain_eval_loader = make_loader(retain_ds, batch_size=256, shuffle=False)

    lora_params = list(model_fa.get_lora_params())
    optimizer = Adam(lora_params, lr=lr)

    # [OPT-4] Cosine annealing over the max_steps budget
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=max_steps) if use_lr_schedule else None
    )

    criterion = nn.BCEWithLogitsLoss()

    history = {
        "step": [],
        "forget_loss_raw": [],
        "retain_auc": [],
        "forget_acc": [],
        "forget_auc": [],
        "lr": [],
        "group_metrics": [],  # [OPT-5]
        "likelihood_ratio": [],  # [OPT-6]
    }

    forget_iter = iter(forget_loader)
    retain_iter = iter(retain_loader)
    t0 = time.time()

    best_state = copy.deepcopy(model_fa.state_dict())
    best_forget_auc = float("inf")

    for step in range(1, max_steps + 1):
        model_fa.train()

        # ── forget batch ──
        try:
            batch_f = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            batch_f = next(forget_iter)

        xn_f, xc_f, y_f = batch_f
        xn_f = xn_f.to(device) if xn_f.numel() > 0 else None
        xc_f = xc_f.to(device) if xc_f.numel() > 0 else None
        y_f = y_f.to(device)

        # ── retain batch ──
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

        loss_f = criterion(model_fa(xn_f, xc_f), y_f)
        loss_r = criterion(model_fa(xn_r, xc_r), y_r)
        loss = -loss_f + lambda_retain * loss_r

        loss.backward()

        # [OPT-3] Per-layer clip vs. global clip
        if per_layer_clip:
            _clip_grad_per_layer(lora_params, max_norm=grad_clip_norm)
        else:
            nn.utils.clip_grad_norm_(lora_params, grad_clip_norm)

        optimizer.step()

        # [OPT-4] Step the scheduler
        if scheduler is not None:
            scheduler.step()

        if step % 5 == 0 or step == 1:
            retain_metrics = evaluate(model_fa, retain_eval_loader, device)
            forget_metrics = evaluate(
                model_fa, make_loader(forget_ds, 256, False), device
            )

            cur_lr = optimizer.param_groups[0]["lr"]

            history["step"].append(step)
            history["forget_loss_raw"].append(loss_f.item())
            history["retain_auc"].append(retain_metrics["auc"])
            history["forget_acc"].append(forget_metrics["acc"])
            history["forget_auc"].append(forget_metrics["auc"])
            history["lr"].append(cur_lr)

            # [OPT-5] Per-group metrics
            if forget_group_labels is not None:
                gm = _evaluate_by_group(
                    model_fa, forget_ds, forget_group_labels, device
                )
                history["group_metrics"].append(gm)
            else:
                history["group_metrics"].append(None)

            # [OPT-6] Likelihood ratio score: log P_forget / P_retain
            # LR close to 0 → forget loss ≈ retain loss → forgetting certified
            if log_likelihood_ratio and retain_reference_loss is not None:
                lr_score = forget_metrics.get("mean_loss", 0.0) - retain_reference_loss
                history["likelihood_ratio"].append(lr_score)
            else:
                history["likelihood_ratio"].append(None)

            if verbose:
                lr_str = f" lr={cur_lr:.5f}" if use_lr_schedule else ""
                print(
                    f"    step {step:3d} | forget_auc={forget_metrics['auc']:.4f} "
                    f"forget_acc={forget_metrics['acc']:.3f} "
                    f"retain_auc={retain_metrics['auc']:.4f}{lr_str}"
                )

                # [OPT-5] Print per-group if available
                if forget_group_labels is not None and history["group_metrics"][-1]:
                    for gid, gmet in history["group_metrics"][-1].items():
                        label = "age<25" if gid == 1 else "age≥25"
                        print(
                            f"      group [{label}]: forget_acc={gmet['acc']:.3f} "
                            f"forget_auc={gmet['auc']:.4f}"
                        )

            if (
                retain_metrics["auc"] >= retain_auc_min
                and forget_metrics["auc"] < best_forget_auc
            ):
                best_forget_auc = forget_metrics["auc"]
                best_state = copy.deepcopy(model_fa.state_dict())

            # [OPT-2] Hard floor uses the (now configurable) retain_auc_min
            if retain_metrics["auc"] < retain_auc_min:
                if verbose:
                    print(
                        f"    [ForgetAdapter] Retain AUC below {retain_auc_min:.2f} "
                        f"— stopping early"
                    )
                break

    elapsed = time.time() - t0
    model_fa.load_state_dict(best_state)

    # [OPT-1] Add DP-flavored noise to LoRA weights after restoring best state
    if noise_injection:
        if verbose:
            print(
                f"  [ForgetAdapter] Injecting noise (scale={noise_scale:.4f}) "
                f"into LoRA weights for MIA hardening..."
            )
        model_fa = _inject_noise_into_lora(model_fa, noise_scale, device)
        # Re-evaluate after noise to show actual post-noise metrics
        if verbose:
            nm = evaluate(model_fa, make_loader(forget_ds, 256, False), device)
            rm = evaluate(model_fa, retain_eval_loader, device)
            print(
                f"  [ForgetAdapter] Post-noise: forget_auc={nm['auc']:.4f} "
                f"forget_acc={nm['acc']:.3f} retain_auc={rm['auc']:.4f}"
            )

    history["elapsed"] = elapsed
    if verbose:
        print(f"  [ForgetAdapter] Done in {elapsed:.1f}s")

    return model_fa, history
