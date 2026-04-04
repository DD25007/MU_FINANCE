"""
Scalability experiment:
  - Repeat on German Credit (1K) and Give Me Some Credit (150K)
  - Show wall-clock speedup scales with dataset size
  - Temporal unlearning experiment (concept drift forgetting)
"""

import sys
import os
import time
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import prepare_datasets
from train import build_model, train_model, evaluate
from unlearning.forget_adapter import run_forget_adapter
from unlearning.retain_adapter import run_retain_adapter
from unlearning.baselines import baseline_full_retrain
from evaluation.metrics import full_evaluation, forget_set_accuracy, compute_auc


def run_scalability_experiment(cfg: dict, device: torch.device) -> dict:
    """
    Runs LoRA unlearning and full-retrain on multiple datasets,
    compares wall-clock time and AUC.
    """
    results = {}

    for dataset_name in ["german", "gmsc"]:
        print(f"\n{'='*50}")
        print(f"Scalability: {dataset_name}")
        print(f"{'='*50}")

        data = prepare_datasets(
            dataset_name=dataset_name,
            data_dir=cfg["data_dir"],
            forget_strategy="random",
            forget_frac=0.10,
            seed=cfg["seed"],
        )

        full_train = data["full_train"]
        val_ds = data["val"]
        test_ds = data["test"]
        forget_ds = data["forget"]
        retain_ds = data["retain"]
        forget_indices = data["forget_indices"]
        cat_dims = data["cat_dims"]
        num_num_features = data["num_num_features"]

        def model_factory():
            return build_model(
                cfg["arch"], num_num_features, cat_dims, device,
                d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                n_layers=cfg["n_layers"], dropout=cfg["dropout"]
            )

        # Train base model
        print("Training base model...")
        t0 = time.time()
        base_model = model_factory()
        base_model, _ = train_model(
            base_model, full_train, val_ds, device,
            epochs=cfg["epochs"], batch_size=cfg["batch_size"],
            lr=cfg["lr"], verbose=False
        )
        base_train_time = time.time() - t0
        base_forget_acc = forget_set_accuracy(base_model, forget_ds, device)

        # LoRA unlearning
        print("Running LoRA unlearning...")
        t_lora_start = time.time()
        m_fa, _ = run_forget_adapter(
            base_model, forget_ds, retain_ds, device,
            lora_r=cfg["lora_rank_default"], max_steps=cfg["fa_steps"], verbose=False
        )
        m_star, _ = run_retain_adapter(
            m_fa, base_model, retain_ds, val_ds, device,
            lora_r=cfg["lora_rank_default"], epochs=cfg["ra_epochs"], verbose=False
        )
        lora_time = time.time() - t_lora_start
        lora_metrics = full_evaluation(
            m_star, base_model, forget_ds, retain_ds, test_ds,
            device, base_forget_acc, elapsed_seconds=lora_time, verbose=True
        )

        # Full retrain
        print("Running full retrain baseline...")
        m_retrain, h_retrain = baseline_full_retrain(
            model_factory, retain_ds, val_ds, device,
            epochs=cfg["epochs"], batch_size=cfg["batch_size"], verbose=False
        )
        retrain_metrics = full_evaluation(
            m_retrain, base_model, forget_ds, retain_ds, test_ds,
            device, base_forget_acc, elapsed_seconds=h_retrain["elapsed"], verbose=True
        )

        speedup = h_retrain["elapsed"] / max(lora_time, 1e-3)
        print(f"\n  Dataset: {dataset_name} | N_train={len(full_train)}")
        print(f"  Base train time: {base_train_time:.1f}s")
        print(f"  LoRA unlearn time: {lora_time:.1f}s")
        print(f"  Full retrain time: {h_retrain['elapsed']:.1f}s")
        print(f"  Speedup: {speedup:.1f}x")

        results[dataset_name] = {
            "n_train": len(full_train),
            "n_forget": len(forget_ds),
            "base_train_time": base_train_time,
            "lora_unlearn_time": lora_time,
            "full_retrain_time": h_retrain["elapsed"],
            "speedup": speedup,
            "lora_metrics": lora_metrics,
            "retrain_metrics": retrain_metrics,
        }

    return results
