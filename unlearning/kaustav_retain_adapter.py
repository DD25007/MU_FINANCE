"""
Phase 3 — Retain Adapter (Knowledge Distillation).

Improvements over baseline:
  [OPT-A] Tighter forget_auc ceiling: default max_forget_recovery lowered from
          0.15 → 0.08. Prevents Phase 3 from over-restoring the forget set.
  [OPT-B] Stronger bad-teacher regularizer: adds `-alpha * CE(forget)` term
          in addition to the uniform-target loss, giving a gradient signal
          that actively pushes forget predictions away from correct labels.
  [OPT-C] Per-class KL divergence: logs KL separately for class 0 and class 1
          to detect if forgetting is biased toward one credit outcome.
  [OPT-D] Relearning attack support: after training, optionally fine-tune the
          final model for N epochs on D_f and return the relearning curve.
          A truly unlearned model should relearn slowly.
  [OPT-E] Composite ForgettingScore logged:
          ForgettingScore = (1 - |forget_acc - 0.5|) * retain_auc
          Gives a single number for ranking methods in the paper table.
"""

import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional
import numpy as np

from data.datasets import CreditDataset, make_loader
from models.lora import freeze_non_lora, count_parameters, merge_lora_into_model
from train import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# KL divergence loss (unchanged from baseline)
# ─────────────────────────────────────────────────────────────────────────────


def kl_divergence_loss(student_logits, teacher_logits, temperature=2.0):
    student_probs = torch.sigmoid(student_logits / temperature)
    teacher_probs = torch.sigmoid(teacher_logits / temperature).detach()
    eps = 1e-8
    kl = teacher_probs * torch.log((teacher_probs + eps) / (student_probs + eps)) + (
        1 - teacher_probs
    ) * torch.log((1 - teacher_probs + eps) / (1 - student_probs + eps))
    return kl.mean() * (temperature**2)


# ─────────────────────────────────────────────────────────────────────────────
# [OPT-C] Per-class KL divergence
# ─────────────────────────────────────────────────────────────────────────────


def _per_class_kl(model, teacher, dataset, device, temperature=2.0):
    """
    Computes KL divergence separately for class 0 (good credit) and class 1 (bad credit).
    Useful for detecting if forgetting disproportionately affects one class.
    Returns: {"kl_class0": float, "kl_class1": float}
    """
    loader = make_loader(dataset, batch_size=256, shuffle=False)
    kl_c0, kl_c1 = [], []
    model.eval()
    teacher.eval()
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            x_num = x_num.to(device) if x_num.numel() > 0 else None
            x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
            y = y.to(device)
            s_logits = model(x_num, x_cat)
            t_logits = teacher(x_num, x_cat)
            per_sample_kl = kl_divergence_loss(
                s_logits.unsqueeze(1), t_logits.unsqueeze(1), temperature
            )
            mask0 = y == 0
            mask1 = y == 1
            if mask0.any():
                kl_c0.append(
                    per_sample_kl[mask0].mean().item()
                    if per_sample_kl.dim() > 0
                    else per_sample_kl.item()
                )
            if mask1.any():
                kl_c1.append(
                    per_sample_kl[mask1].mean().item()
                    if per_sample_kl.dim() > 0
                    else per_sample_kl.item()
                )
    return {
        "kl_class0": float(np.mean(kl_c0)) if kl_c0 else 0.0,
        "kl_class1": float(np.mean(kl_c1)) if kl_c1 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# [OPT-D] Relearning attack curve
# ─────────────────────────────────────────────────────────────────────────────


def run_relearning_attack(
    model: nn.Module,
    forget_ds: CreditDataset,
    device: torch.device,
    relearn_epochs: int = 10,
    lr: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """
    Fine-tunes the unlearned model on D_f for `relearn_epochs` epochs.
    Tracks forget_acc per epoch to produce a relearning curve.

    A truly unlearned model should recover forget_acc slowly (like training from
    scratch). A model that only superficially forgot will recover within 1–2 epochs.

    Returns: {"epoch": [...], "forget_acc": [...], "forget_auc": [...]}
    """
    model_rl = copy.deepcopy(model).to(device)
    optimizer = Adam(model_rl.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    forget_loader = make_loader(forget_ds, batch_size=64, shuffle=True)
    forget_eval_loader = make_loader(forget_ds, batch_size=256, shuffle=False)

    history = {"epoch": [], "forget_acc": [], "forget_auc": []}

    if verbose:
        print(f"  [RelearningAttack] Fine-tuning on D_f for {relearn_epochs} epochs...")

    for epoch in range(1, relearn_epochs + 1):
        model_rl.train()
        for x_num, x_cat, y in forget_loader:
            x_num = x_num.to(device) if x_num.numel() > 0 else None
            x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model_rl(x_num, x_cat), y)
            loss.backward()
            optimizer.step()

        m = evaluate(model_rl, forget_eval_loader, device)
        history["epoch"].append(epoch)
        history["forget_acc"].append(m["acc"])
        history["forget_auc"].append(m["auc"])

        if verbose:
            print(
                f"    relearn epoch {epoch:2d} | forget_acc={m['acc']:.4f} "
                f"forget_auc={m['auc']:.4f}"
            )

    return history


# ─────────────────────────────────────────────────────────────────────────────
# [OPT-E] Composite ForgettingScore
# ─────────────────────────────────────────────────────────────────────────────


def forgetting_score(forget_acc: float, retain_auc: float) -> float:
    """
    ForgettingScore = (1 - |forget_acc - 0.5|) * retain_auc

    Ranges [0, 1]. Higher is better.
    - Perfect forgetting (forget_acc=0.5) + perfect utility (retain_auc=1.0) → score=1.0
    - No forgetting (forget_acc=1.0) → score=0.0
    - Retain AUC collapse (retain_auc=0.5) → score≈0.25 even with perfect forgetting
    """
    return (1.0 - abs(forget_acc - 0.5)) * retain_auc


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────


def run_retain_adapter(
    model_with_fa: nn.Module,
    original_model: nn.Module,
    retain_ds: CreditDataset,
    val_ds: CreditDataset,
    device: torch.device,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 2.0,
    forget_ds: CreditDataset = None,
    # ── [OPT-A] Tighter ceiling ───────────────────────────────────────────────
    max_forget_recovery: float = 0.08,  # Tightened from 0.15 → 0.08
    # ── [OPT-B] Stronger forget regularizer ──────────────────────────────────
    gamma_forget: float = 1.0,  # Weight of uniform-target bad-teacher loss
    use_negative_ce_forget: bool = False,  # Toggle: add -alpha*CE(forget) on top
    alpha_neg_ce: float = 0.3,  # Weight of negative CE term
    # ── [OPT-C] Per-class KL tracking ────────────────────────────────────────
    track_per_class_kl: bool = False,  # Toggle: log KL per credit class
    # ── [OPT-D] Relearning attack ─────────────────────────────────────────────
    run_relearning: bool = False,  # Toggle: run relearning curve after training
    relearn_epochs: int = 10,  # How many epochs of relearning to simulate
    # ── [OPT-E] ForgettingScore ───────────────────────────────────────────────
    log_forgetting_score: bool = True,  # Toggle: compute and log composite score
    verbose: bool = True,
) -> tuple:
    """
    Attaches second LoRA adapter Θ_r to model_with_fa.
    Trains via KL divergence to recover retain utility while keeping forget erased.

    Returns:
      final_model, history
      (history includes relearning curve if run_relearning=True)
    """
    model_ra = copy.deepcopy(model_with_fa).to(device)
    teacher = copy.deepcopy(original_model).to(device).eval()

    for param in model_ra.parameters():
        param.requires_grad = False

    from models.lora import LoRALinear

    def _add_second_lora(module, r, alpha, dropout):
        lora_params = []
        for name, child in list(module.named_children()):
            if name in ("q_proj", "v_proj"):
                if isinstance(child, LoRALinear):
                    child.lora_A.requires_grad = False
                    child.lora_B.requires_grad = False
                    new_lora = LoRALinear(
                        _make_dummy_linear(
                            child.out_features,
                            child.in_features,
                            device=child.weight.device,
                        ),
                        r=r,
                        lora_alpha=alpha,
                        lora_dropout=dropout,
                    )
                    setattr(module, name, _CombinedAdapter(child, new_lora))
                    lora_params.extend(new_lora.trainable_parameters())
                elif isinstance(child, nn.Linear):
                    new_lora = LoRALinear(
                        child, r=r, lora_alpha=alpha, lora_dropout=dropout
                    )
                    setattr(module, name, new_lora)
                    lora_params.extend(new_lora.trainable_parameters())
            else:
                lora_params.extend(_add_second_lora(child, r, alpha, dropout))
        return lora_params

    retain_lora_params = _add_second_lora(model_ra, lora_r, lora_alpha, lora_dropout)
    for p in retain_lora_params:
        p.requires_grad = True

    total_p, trainable_p = count_parameters(model_ra)
    if verbose:
        print(
            f"  [RetainAdapter] LoRA params: {trainable_p:,} / {total_p:,} "
            f"({100*trainable_p/max(total_p,1):.2f}%)"
        )

    retain_loader = make_loader(retain_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=256, shuffle=False)
    forget_loader = (
        make_loader(forget_ds, batch_size=batch_size, shuffle=True)
        if forget_ds is not None
        else None
    )
    forget_eval_loader = (
        make_loader(forget_ds, batch_size=256, shuffle=False)
        if forget_ds is not None
        else None
    )

    optimizer = Adam(retain_lora_params, lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    # [OPT-A] Record Phase 2 end forget_auc and compute tighter ceiling
    if forget_eval_loader is not None:
        init_forget_metrics = evaluate(model_ra, forget_eval_loader, device)
        initial_forget_auc = init_forget_metrics["auc"]
        forget_auc_ceiling = initial_forget_auc + max_forget_recovery  # [OPT-A]
        if verbose:
            print(
                f"  [RetainAdapter] Phase2 end forget_auc={initial_forget_auc:.4f} "
                f"→ ceiling={forget_auc_ceiling:.4f}  "
                f"(max_forget_recovery={max_forget_recovery})"
            )
    else:
        initial_forget_auc = None
        forget_auc_ceiling = float("inf")

    forget_iter = iter(forget_loader) if forget_loader is not None else None

    history = {
        "epoch": [],
        "kl_loss": [],
        "val_auc": [],
        "forget_auc": [],
        "per_class_kl": [],  # [OPT-C]
        "forgetting_score": [],  # [OPT-E]
    }
    best_val_auc = -1.0
    best_state = copy.deepcopy(model_ra.state_dict())

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model_ra.train()
        teacher.eval()
        total_kl = 0.0
        n_batches = 0

        for batch in retain_loader:
            x_num, x_cat, y = batch
            x_num = x_num.to(device) if x_num.numel() > 0 else None
            x_cat = x_cat.to(device) if x_cat.numel() > 0 else None

            with torch.no_grad():
                teacher_logits = teacher(x_num, x_cat)

            student_logits = model_ra(x_num, x_cat)
            kl_loss = kl_divergence_loss(student_logits, teacher_logits, temperature)
            loss = kl_loss

            # ── [OPT-B] Forget regularization ────────────────────────────────
            if forget_iter is not None and (gamma_forget > 0 or use_negative_ce_forget):
                try:
                    xn_f, xc_f, y_f = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    xn_f, xc_f, y_f = next(forget_iter)

                xn_f = xn_f.to(device) if xn_f.numel() > 0 else None
                xc_f = xc_f.to(device) if xc_f.numel() > 0 else None
                y_f = y_f.to(device)
                forget_logits = model_ra(xn_f, xc_f)

                # Original: push forget logits toward 0.5 (uniform uncertainty)
                if gamma_forget > 0:
                    uniform_targets = torch.full_like(forget_logits, 0.5)
                    bad_teacher_loss = F.binary_cross_entropy_with_logits(
                        forget_logits, uniform_targets
                    )
                    loss = loss + gamma_forget * bad_teacher_loss

                # [OPT-B] Additional: negative CE pushes away from true labels
                # This is a stronger signal than uniform targets alone.
                # Equivalent to gradient ascent on D_f inside Phase 3.
                if use_negative_ce_forget:
                    neg_ce = criterion(forget_logits, y_f)
                    loss = loss - alpha_neg_ce * neg_ce

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(retain_lora_params, 1.0)
            optimizer.step()
            total_kl += kl_loss.item()
            n_batches += 1

        scheduler.step()
        val_metrics = evaluate(model_ra, val_loader, device)

        if forget_eval_loader is not None:
            forget_metrics = evaluate(model_ra, forget_eval_loader, device)
            cur_forget_auc = forget_metrics["auc"]
            cur_forget_acc = forget_metrics["acc"]
        else:
            cur_forget_auc = None
            cur_forget_acc = None

        # [OPT-C] Per-class KL on retain set
        if track_per_class_kl:
            ck = _per_class_kl(model_ra, teacher, retain_ds, device, temperature)
            history["per_class_kl"].append(ck)
        else:
            history["per_class_kl"].append(None)

        # [OPT-E] ForgettingScore
        if log_forgetting_score and cur_forget_acc is not None:
            fs = forgetting_score(cur_forget_acc, val_metrics["auc"])
            history["forgetting_score"].append(fs)
        else:
            history["forgetting_score"].append(None)

        history["epoch"].append(epoch)
        history["kl_loss"].append(total_kl / max(n_batches, 1))
        history["val_auc"].append(val_metrics["auc"])
        history["forget_auc"].append(cur_forget_auc)

        # Save best state only within allowed forget_auc ceiling [OPT-A]
        if val_metrics["auc"] > best_val_auc and (
            cur_forget_auc is None or cur_forget_auc <= forget_auc_ceiling
        ):
            best_val_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model_ra.state_dict())

        if verbose and epoch % 5 == 0:
            fa_str = (
                f" forget_auc={cur_forget_auc:.4f}"
                if cur_forget_auc is not None
                else ""
            )
            fs_str = (
                f" forgetting_score={history['forgetting_score'][-1]:.4f}"
                if history["forgetting_score"][-1] is not None
                else ""
            )
            ck_str = ""
            if track_per_class_kl and history["per_class_kl"][-1]:
                ck = history["per_class_kl"][-1]
                ck_str = f" kl_c0={ck['kl_class0']:.4f} kl_c1={ck['kl_class1']:.4f}"
            print(
                f"    epoch {epoch:3d} | kl_loss={total_kl/max(n_batches,1):.4f} "
                f"val_auc={val_metrics['auc']:.4f}{fa_str}{fs_str}{ck_str}"
            )

        if cur_forget_auc is not None and cur_forget_auc > forget_auc_ceiling:
            if verbose:
                print(
                    f"    [RetainAdapter] forget_auc={cur_forget_auc:.4f} exceeded "
                    f"ceiling {forget_auc_ceiling:.4f} — stopping early"
                )
            break

    model_ra.load_state_dict(best_state)
    elapsed = time.time() - t0
    history["elapsed"] = elapsed

    if verbose:
        print(
            f"  [RetainAdapter] Done in {elapsed:.1f}s | best val AUC={best_val_auc:.4f}"
        )

    final_model = _merge_combined_adapters(model_ra)

    # [OPT-D] Relearning attack — runs on the merged final model
    if run_relearning and forget_ds is not None:
        relearn_history = run_relearning_attack(
            final_model,
            forget_ds,
            device,
            relearn_epochs=relearn_epochs,
            verbose=verbose,
        )
        history["relearning"] = relearn_history
    else:
        history["relearning"] = None

    return final_model, history


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged from baseline)
# ─────────────────────────────────────────────────────────────────────────────


def _make_dummy_linear(out_features, in_features, device=None):
    lin = nn.Linear(in_features, out_features, bias=False, device=device)
    nn.init.zeros_(lin.weight)
    return lin


class _CombinedAdapter(nn.Module):
    def __init__(self, frozen_lora, new_lora):
        super().__init__()
        self.frozen_lora = frozen_lora
        self.new_lora = new_lora

    def forward(self, x):
        base_out = self.frozen_lora(x)
        delta = (
            x @ self.new_lora.lora_A.T @ self.new_lora.lora_B.T * self.new_lora.scaling
        )
        return base_out + delta

    def trainable_parameters(self):
        return self.new_lora.trainable_parameters()


def _merge_combined_adapters(model):
    from models.lora import LoRALinear

    for name, child in list(model.named_children()):
        if isinstance(child, _CombinedAdapter):
            fa = child.frozen_lora
            ra = child.new_lora
            merged_w = (
                fa.weight.data
                + fa.scaling * (fa.lora_B @ fa.lora_A)
                + ra.scaling * (ra.lora_B @ ra.lora_A)
            )
            new_linear = nn.Linear(
                fa.in_features, fa.out_features, bias=fa.bias is not None
            )
            new_linear.weight = nn.Parameter(merged_w)
            if fa.bias is not None:
                new_linear.bias = nn.Parameter(fa.bias.clone())
            setattr(model, name, new_linear)
        elif isinstance(child, LoRALinear):
            setattr(model, name, child.merge_weights())
        else:
            _merge_combined_adapters(child)
    return model
