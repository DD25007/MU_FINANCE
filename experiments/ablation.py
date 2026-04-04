"""
Ablation study: systematically removes components to prove each is necessary.

Ablations:
  A1: Phase 2 only (no retain adapter) — does forgetting hold without Phase 3?
  A2: Phase 3 only (no forget adapter) — does the model forget without Phase 2?
  A3: Vary LoRA rank r ∈ {4, 8, 16} — sensitivity to rank
  A4: Vary forget-set size |D_f| ∈ {5%, 10%, 20%} — scalability within method
  A5: Vary gradient ascent steps — sensitivity to step count
"""

import sys
import os
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import prepare_datasets
from train import build_model, train_model, load_model, evaluate
from unlearning.forget_adapter import run_forget_adapter
from unlearning.retain_adapter import run_retain_adapter
from evaluation.metrics import full_evaluation, forget_set_accuracy, compute_auc
from evaluation.mia import loss_based_mia


def run_ablation_study(cfg: dict, base_model, data: dict, device: torch.device) -> dict:
    """
    Run all ablation experiments. Returns dict of results.
    """
    forget_ds = data["forget"]
    retain_ds = data["retain"]
    val_ds = data["val"]
    test_ds = data["test"]
    base_forget_acc = forget_set_accuracy(base_model, forget_ds, device)

    results = {}

    # ── A1: Phase 2 only (no Phase 3) ──────────────────────────────────
    print("\n[Ablation A1] Phase 2 only (no retain adapter)")
    m_fa, _ = run_forget_adapter(
        base_model, forget_ds, retain_ds, device,
        lora_r=cfg["lora_rank_default"], max_steps=cfg["fa_steps"], verbose=False
    )
    r_a1 = full_evaluation(m_fa, base_model, forget_ds, retain_ds, test_ds,
                           device, base_forget_acc, verbose=True)
    mia_a1 = loss_based_mia(m_fa, forget_ds, retain_ds, device, verbose=False)
    r_a1["mia_score"] = mia_a1["mia_score"]
    results["phase2_only"] = r_a1

    # ── A2: Phase 3 only (no Phase 2) ──────────────────────────────────
    print("\n[Ablation A2] Phase 3 only (no forget adapter)")
    m_ra, _ = run_retain_adapter(
        base_model, base_model, retain_ds, val_ds, device,
        lora_r=cfg["lora_rank_default"], epochs=cfg["ra_epochs"], verbose=False
    )
    r_a2 = full_evaluation(m_ra, base_model, forget_ds, retain_ds, test_ds,
                           device, base_forget_acc, verbose=True)
    mia_a2 = loss_based_mia(m_ra, forget_ds, retain_ds, device, verbose=False)
    r_a2["mia_score"] = mia_a2["mia_score"]
    results["phase3_only"] = r_a2

    # ── A3: Vary LoRA rank ──────────────────────────────────────────────
    print("\n[Ablation A3] Vary LoRA rank")
    rank_results = {}
    for rank in cfg["lora_ranks"]:
        print(f"  rank={rank}")
        m_fa_r, _ = run_forget_adapter(
            base_model, forget_ds, retain_ds, device,
            lora_r=rank, max_steps=cfg["fa_steps"], verbose=False
        )
        m_star_r, _ = run_retain_adapter(
            m_fa_r, base_model, retain_ds, val_ds, device,
            lora_r=rank, epochs=cfg["ra_epochs"], verbose=False
        )
        r = full_evaluation(m_star_r, base_model, forget_ds, retain_ds, test_ds,
                            device, base_forget_acc, verbose=True)
        mia_r = loss_based_mia(m_star_r, forget_ds, retain_ds, device, verbose=False)
        r["mia_score"] = mia_r["mia_score"]
        r["rank"] = rank
        rank_results[f"rank_{rank}"] = r
    results["vary_rank"] = rank_results

    # ── A4: Vary gradient ascent steps ─────────────────────────────────
    print("\n[Ablation A4] Vary gradient ascent steps")
    step_results = {}
    rank = cfg["lora_rank_default"]
    for n_steps in [10, 25, 50, 100]:
        print(f"  steps={n_steps}")
        m_fa_s, _ = run_forget_adapter(
            base_model, forget_ds, retain_ds, device,
            lora_r=rank, max_steps=n_steps, verbose=False
        )
        m_star_s, _ = run_retain_adapter(
            m_fa_s, base_model, retain_ds, val_ds, device,
            lora_r=rank, epochs=cfg["ra_epochs"], verbose=False
        )
        r = full_evaluation(m_star_s, base_model, forget_ds, retain_ds, test_ds,
                            device, base_forget_acc, verbose=True)
        r["steps"] = n_steps
        step_results[f"steps_{n_steps}"] = r
    results["vary_steps"] = step_results

    return results
