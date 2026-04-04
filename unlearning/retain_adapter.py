"""
Phase 3 — Retain Adapter (Knowledge Distillation).

Attaches a SECOND LoRA adapter Θ_r to M+Θ_f.
Trains Θ_r to match the ORIGINAL frozen model M's predictions on D_r via KL divergence.
This repairs any collateral damage from the forget adapter while keeping forget set erased.

After training: merge Θ_r into the model and discard Θ_f → final M*.
"""

import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.datasets import CreditDataset, make_loader
from models.lora import freeze_non_lora, count_parameters, merge_lora_into_model
from train import evaluate


def kl_divergence_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                       temperature: float = 2.0) -> torch.Tensor:
    """
    Soft KL divergence between student and teacher for binary classification.
    Uses temperature scaling for softer targets.
    """
    student_probs = torch.sigmoid(student_logits / temperature)
    teacher_probs = torch.sigmoid(teacher_logits / temperature).detach()

    # Binary KL: KL(teacher ∥ student)
    eps = 1e-8
    kl = teacher_probs * torch.log((teacher_probs + eps) / (student_probs + eps)) + \
         (1 - teacher_probs) * torch.log((1 - teacher_probs + eps) / (1 - student_probs + eps))
    return kl.mean() * (temperature ** 2)


def run_retain_adapter(
    model_with_fa: nn.Module,        # M + Θ_f (output of Phase 2)
    original_model: nn.Module,       # M (original frozen base model)
    retain_ds: CreditDataset,
    val_ds: CreditDataset,
    device: torch.device,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 2.0,
    forget_ds: CreditDataset = None,
    max_forget_recovery: float = 0.05,   # stop if forget_auc rises more than this above Phase 2 end
    gamma_forget: float = 1.0,           # weight of bad-teacher loss on D_f (0 = disabled)
    verbose: bool = True,
) -> tuple:
    """
    Attaches a second LoRA adapter Θ_r to model_with_fa.
    Trains Θ_r to minimise KL(M+Θ_f+Θ_r(x) ∥ M(x)) on D_r.

    If forget_ds is provided:
      - Adds a "bad teacher" loss term that pushes D_f predictions toward
        uniform uncertainty (0.5) — actively prevents forget-set restoration.
      - Monitors forget_auc each epoch and stops early if it rises more than
        max_forget_recovery above the Phase 2 end value.
    Best state = best val_auc where forget_auc <= initial + max_forget_recovery.

    Merges Θ_r into weights to produce M*.

    Returns:
      final_model  — M* with Θ_r merged in, Θ_f discarded
      history
    """
    # Work on a copy
    model_ra = copy.deepcopy(model_with_fa).to(device)
    teacher = copy.deepcopy(original_model).to(device).eval()

    # Freeze everything first (including Θ_f from Phase 2)
    for param in model_ra.parameters():
        param.requires_grad = False

    # Attach SECOND LoRA adapter on top (replaces Q/V projections again with new adapters)
    # We need to be careful: q_proj/v_proj are already LoRALinear from Phase 2.
    # Strategy: create new LoRALinear wrapping the existing LoRALinear.
    # Instead, we add a separate adapter layer that adds on top.
    from models.lora import LoRALinear

    def _add_second_lora(module, r, alpha, dropout):
        """Wrap existing LoRALinear or nn.Linear with another LoRALinear."""
        lora_params = []
        for name, child in list(module.named_children()):
            if name in ("q_proj", "v_proj"):
                # Create a new LoRALinear wrapping the frozen existing layer
                if isinstance(child, LoRALinear):
                    # Freeze the existing LoRA
                    child.lora_A.requires_grad = False
                    child.lora_B.requires_grad = False
                    # We'll add a residual adapter
                    new_lora = LoRALinear(
                        _make_dummy_linear(child.out_features, child.in_features,
                                           device=child.weight.device),
                        r=r, lora_alpha=alpha, lora_dropout=dropout
                    )
                    # Wrap in a sequential combiner
                    setattr(module, name, _CombinedAdapter(child, new_lora))
                    lora_params.extend(new_lora.trainable_parameters())
                elif isinstance(child, nn.Linear):
                    new_lora = LoRALinear(child, r=r, lora_alpha=alpha, lora_dropout=dropout)
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
        print(f"  [RetainAdapter] LoRA params: {trainable_p:,} / {total_p:,} "
              f"({100*trainable_p/max(total_p,1):.2f}%)")

    retain_loader = make_loader(retain_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=256, shuffle=False)
    forget_loader = make_loader(forget_ds, batch_size=batch_size, shuffle=True) if forget_ds is not None else None
    forget_eval_loader = make_loader(forget_ds, batch_size=256, shuffle=False) if forget_ds is not None else None

    optimizer = Adam(retain_lora_params, lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Record forget_auc at Phase 2 end (before any Phase 3 training)
    if forget_eval_loader is not None:
        init_forget_metrics = evaluate(model_ra, forget_eval_loader, device)
        initial_forget_auc = init_forget_metrics["auc"]
        forget_auc_ceiling = initial_forget_auc + max_forget_recovery
        if verbose:
            print(f"  [RetainAdapter] Phase2 end forget_auc={initial_forget_auc:.4f} "
                  f"→ ceiling={forget_auc_ceiling:.4f}")
    else:
        initial_forget_auc = None
        forget_auc_ceiling = float("inf")

    # Iterator for bad-teacher batches (shuffled, same size as retain batches)
    forget_iter = iter(forget_loader) if forget_loader is not None else None

    history = {"epoch": [], "kl_loss": [], "val_auc": [], "forget_auc": []}
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

            # Bad-teacher loss: push D_f logits toward zero (σ(0)=0.5 = maximum
            # uncertainty). With a high-enough gamma, all D_f logits collapse to
            # near-zero, eliminating inter-sample variation → AUC → 0.5.
            if forget_iter is not None and gamma_forget > 0:
                try:
                    xn_f, xc_f, _ = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    xn_f, xc_f, _ = next(forget_iter)
                xn_f = xn_f.to(device) if xn_f.numel() > 0 else None
                xc_f = xc_f.to(device) if xc_f.numel() > 0 else None
                forget_logits = model_ra(xn_f, xc_f)
                uniform_targets = torch.full_like(forget_logits, 0.5)
                bad_teacher_loss = F.binary_cross_entropy_with_logits(
                    forget_logits, uniform_targets)
                loss = kl_loss + gamma_forget * bad_teacher_loss
            else:
                loss = kl_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(retain_lora_params, 1.0)
            optimizer.step()
            total_kl += kl_loss.item()
            n_batches += 1

        scheduler.step()
        val_metrics = evaluate(model_ra, val_loader, device)

        # Monitor forget_auc to detect restoration of forgotten knowledge
        if forget_eval_loader is not None:
            forget_metrics = evaluate(model_ra, forget_eval_loader, device)
            cur_forget_auc = forget_metrics["auc"]
        else:
            cur_forget_auc = None

        history["epoch"].append(epoch)
        history["kl_loss"].append(total_kl / max(n_batches, 1))
        history["val_auc"].append(val_metrics["auc"])
        history["forget_auc"].append(cur_forget_auc)

        # Save best state only while forget_auc is within allowed recovery
        if val_metrics["auc"] > best_val_auc and (cur_forget_auc is None or
                                                   cur_forget_auc <= forget_auc_ceiling):
            best_val_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model_ra.state_dict())

        if verbose and epoch % 5 == 0:
            fa_str = f" forget_auc={cur_forget_auc:.4f}" if cur_forget_auc is not None else ""
            print(f"    epoch {epoch:3d} | kl_loss={total_kl/max(n_batches,1):.4f} "
                  f"val_auc={val_metrics['auc']:.4f}{fa_str}")

        # Early stop if forget_auc has risen too far above Phase 2 end value
        if cur_forget_auc is not None and cur_forget_auc > forget_auc_ceiling:
            if verbose:
                print(f"    [RetainAdapter] forget_auc={cur_forget_auc:.4f} exceeded ceiling "
                      f"{forget_auc_ceiling:.4f} — stopping early")
            break

    model_ra.load_state_dict(best_state)
    elapsed = time.time() - t0
    history["elapsed"] = elapsed

    if verbose:
        print(f"  [RetainAdapter] Done in {elapsed:.1f}s | best val AUC={best_val_auc:.4f}")

    # Merge Θ_r into weights → produce M*
    final_model = _merge_combined_adapters(model_ra)
    return final_model, history


def _make_dummy_linear(out_features: int, in_features: int, device=None) -> nn.Linear:
    """Create a zero-weight linear for wrapping inside LoRALinear."""
    lin = nn.Linear(in_features, out_features, bias=False, device=device)
    nn.init.zeros_(lin.weight)
    return lin


class _CombinedAdapter(nn.Module):
    """Combines a frozen LoRALinear (Phase 2) with a new LoRALinear (Phase 3)."""
    def __init__(self, frozen_lora, new_lora):
        super().__init__()
        self.frozen_lora = frozen_lora
        self.new_lora = new_lora

    def forward(self, x):
        base_out = self.frozen_lora(x)
        # new_lora computes residual correction on the input
        delta = x @ self.new_lora.lora_A.T @ self.new_lora.lora_B.T * self.new_lora.scaling
        return base_out + delta

    def trainable_parameters(self):
        return self.new_lora.trainable_parameters()


def _merge_combined_adapters(model: nn.Module) -> nn.Module:
    """
    Traverse model; for _CombinedAdapter layers, merge the retain LoRA into the frozen LoRA.
    For standalone LoRALinear (Phase 2 only), keep as-is but freeze.
    """
    from models.lora import LoRALinear
    for name, child in list(model.named_children()):
        if isinstance(child, _CombinedAdapter):
            # Merge the retain (new_lora) delta into the frozen_lora weight
            fa = child.frozen_lora   # LoRALinear with frozen base + Θ_f
            ra = child.new_lora      # LoRALinear with Θ_r

            # Merged weight = W + Θ_f + Θ_r
            merged_w = fa.weight.data + fa.scaling * (fa.lora_B @ fa.lora_A) \
                                      + ra.scaling * (ra.lora_B @ ra.lora_A)
            new_linear = nn.Linear(fa.in_features, fa.out_features,
                                   bias=fa.bias is not None)
            new_linear.weight = nn.Parameter(merged_w)
            if fa.bias is not None:
                new_linear.bias = nn.Parameter(fa.bias.clone())
            setattr(model, name, new_linear)
        elif isinstance(child, LoRALinear):
            # Phase 2 only (no retain adapter on top) — merge Θ_f
            setattr(model, name, child.merge_weights())
        else:
            _merge_combined_adapters(child)
    return model
