"""
Method 7: SCRUB (Bonus).

Paper: Kurmanji et al., "Towards Unbounded Machine Unlearning" (NeurIPS 2023)

SCRUB uses a teacher-student setup:
  - Student minimizes KL divergence with teacher on D_retain (match teacher)
  - Student maximizes KL divergence with teacher on D_forget (diverge from teacher)

The teacher is the original trained model M (frozen).
The student is a copy that gets updated.
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
    SCRUB unlearning with teacher-student KL divergence.

    config keys:
        lr: float (default 1e-5)
        alpha: float (weight on forget divergence, default 1.0)
        max_steps: int (default 200)
        batch_size: int (default 64)
        retain_auc_threshold: float (default 0.68)
    """
    lr = config.get("lr", 1e-5)
    alpha = config.get("alpha", 1.0)
    max_steps = config.get("max_steps", 200)
    batch_size = config.get("batch_size", 64)
    retain_auc_threshold = config.get("retain_auc_threshold", 0.68)
    verbose = config.get("verbose", True)

    # Teacher = original model (frozen)
    teacher_model = copy.deepcopy(model).to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    # Student = copy that gets updated
    student_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    forget_loader = make_loader(D_forget, batch_size=batch_size, shuffle=True)
    retain_loader = make_loader(D_retain, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(D_val, batch_size=256, shuffle=False)

    forget_iter = cycle(iter(forget_loader))
    retain_iter = cycle(iter(retain_loader))

    student_model.train()
    for step in range(1, max_steps + 1):
        # ---- Retain step: student matches teacher ----
        xr_num, xr_cat, yr = next(retain_iter)
        xr_num = xr_num.to(device) if xr_num.numel() > 0 else None
        xr_cat = xr_cat.to(device) if xr_cat.numel() > 0 else None

        with torch.no_grad():
            teacher_retain_logits = teacher_model(xr_num, xr_cat)
            teacher_retain_probs = torch.sigmoid(teacher_retain_logits)

        student_retain_logits = student_model(xr_num, xr_cat)
        student_retain_probs = torch.sigmoid(student_retain_logits)

        # KL(student || teacher) on retain -- minimize
        # Build 2-class distributions [1-p, p]
        eps = 1e-8
        student_dist = torch.stack([1 - student_retain_probs + eps, student_retain_probs + eps], dim=-1)
        teacher_dist = torch.stack([1 - teacher_retain_probs + eps, teacher_retain_probs + eps], dim=-1)
        retain_kl = kl_loss(student_dist.log(), teacher_dist)

        # ---- Forget step: student diverges from teacher ----
        xf_num, xf_cat, yf = next(forget_iter)
        xf_num = xf_num.to(device) if xf_num.numel() > 0 else None
        xf_cat = xf_cat.to(device) if xf_cat.numel() > 0 else None

        with torch.no_grad():
            teacher_forget_logits = teacher_model(xf_num, xf_cat)
            teacher_forget_probs = torch.sigmoid(teacher_forget_logits)

        student_forget_logits = student_model(xf_num, xf_cat)
        student_forget_probs = torch.sigmoid(student_forget_logits)

        student_f_dist = torch.stack([1 - student_forget_probs + eps, student_forget_probs + eps], dim=-1)
        teacher_f_dist = torch.stack([1 - teacher_forget_probs + eps, teacher_forget_probs + eps], dim=-1)
        forget_kl = kl_loss(student_f_dist.log(), teacher_f_dist)

        # Total loss: minimize retain KL, maximize forget KL
        loss = retain_kl - alpha * forget_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Early stopping
        if step % 20 == 0:
            val_metrics = evaluate(student_model, val_loader, device)
            val_auc = val_metrics["auc"]

            if verbose and step % 50 == 0:
                print(f"    SCRUB step {step}/{max_steps} | val_auc={val_auc:.4f} "
                      f"| retain_kl={retain_kl.item():.4f} forget_kl={forget_kl.item():.4f}")

            if val_auc < retain_auc_threshold:
                if verbose:
                    print(f"    SCRUB: val AUC dropped to {val_auc:.4f} at step {step} -- stopping")
                break

    return student_model


def scrub_unlearn(model, D_forget, D_retain, D_val, config, device):
    """Convenience wrapper. Returns (model, elapsed_seconds)."""
    start = time.perf_counter()
    result = unlearn(model, D_forget, D_retain, D_val, config, device)
    elapsed = time.perf_counter() - start
    return result, elapsed
