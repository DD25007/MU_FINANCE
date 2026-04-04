"""
Data loading, preprocessing, split construction, and forget-set generation
for credit scoring unlearning experiments.

Supported datasets:
  - German Credit (UCI)          -- 1,000 samples, 20 features
  - Give Me Some Credit (Kaggle) -- 150,000 samples, 10 features

Forget-set strategies:
  1. Random    -- random fraction of training data
  2. Demographic -- age-based subgroup (age < 25)
  3. Temporal  -- simulated temporal window (index-based)

All splits are saved to data/processed/ so every method uses identical data.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.datasets import load_german_credit, load_give_me_some_credit, prepare_datasets


# ──────────────────────────────────────────────
# German Credit (UCI)
# ──────────────────────────────────────────────

def load_german_credit_raw(path="data/raw/german.data"):
    """
    Load German Credit from UCI.
    1000 samples, 20 features, binary label (1=good, 2=bad -> remap to 0=good, 1=bad).
    """
    col_names = [f"f{i}" for i in range(20)] + ["label"]
    if os.path.exists(path):
        df = pd.read_csv(path, sep=" ", header=None, names=col_names)
        df["label"] = (df["label"] == 2).astype(int)  # 1 = bad credit (positive class)
    else:
        # Fall back to the existing synthetic generator
        df_raw = load_german_credit(os.path.dirname(path))
        df_raw = df_raw.rename(columns={"target": "label"})
        return df_raw

    # Categorical cols: f0,f2,f3,f5,f6,f8,f9,f11,f13,f14,f16,f18,f19
    # Numerical cols: f1,f4,f7,f10,f12,f15,f17
    # NOTE: f13 is age (numerical, used for demographic forget-set)
    cat_cols = [f"f{i}" for i in [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]]
    num_cols = [f"f{i}" for i in [1, 4, 7, 10, 12, 15, 17]]
    return df, cat_cols, num_cols


# ──────────────────────────────────────────────
# GiveMeSomeCredit (Kaggle)
# ──────────────────────────────────────────────

def load_gmsc_raw(path="data/raw/cs-training.csv"):
    """
    Load GiveMeSomeCredit from Kaggle.
    ~150,000 rows, 10 features, label = SeriousDlqin2yrs.
    """
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        df = df.rename(columns={"SeriousDlqin2yrs": "label"})
        num_cols = [c for c in df.columns if c != "label"]
        return df, [], num_cols  # all numerical, no categoricals
    else:
        df_raw = load_give_me_some_credit(os.path.dirname(path))
        df_raw = df_raw.rename(columns={"target": "label"})
        num_cols = [c for c in df_raw.columns if c != "label"]
        return df_raw, [], num_cols


def preprocess(df, cat_cols, num_cols):
    """One-hot encode categoricals and standardize numericals."""
    df_enc = df.copy()
    if cat_cols:
        df_enc = pd.get_dummies(df_enc, columns=cat_cols)
    scaler = StandardScaler()
    if num_cols:
        df_enc[num_cols] = scaler.fit_transform(df_enc[num_cols])
    return df_enc, scaler


# ──────────────────────────────────────────────
# Splits and Forget Set Construction
# ──────────────────────────────────────────────

def make_splits(df, forget_strategy="random", forget_frac=0.1, seed=42):
    """
    Returns: train_df, val_df, test_df, forget_df, retain_df
    forget_df is a subset of train_df, retain_df = train_df \\ forget_df.
    """
    # 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"]
    )

    # Construct forget set from train only
    if forget_strategy == "random":
        forget_df = train_df.sample(frac=forget_frac, random_state=seed)
    elif forget_strategy == "demographic":
        # German Credit: forget all samples where age < 25
        # Try both column naming conventions
        age_col = None
        for candidate in ["f13", "age"]:
            if candidate in train_df.columns:
                age_col = candidate
                break
        if age_col is not None:
            forget_df = train_df[train_df[age_col] < 25]
            if len(forget_df) == 0:
                print("[Warn] No samples with age<25; falling back to random 10%")
                forget_df = train_df.sample(frac=0.10, random_state=seed)
        else:
            print("[Warn] No age column found; falling back to random")
            forget_df = train_df.sample(frac=forget_frac, random_state=seed)
    elif forget_strategy == "temporal":
        # Simulate temporal window with index range
        cutoff = int(len(train_df) * forget_frac)
        forget_df = train_df.iloc[:cutoff]
    else:
        raise ValueError(f"Unknown forget strategy: {forget_strategy}")

    retain_df = train_df.drop(forget_df.index)
    return train_df, val_df, test_df, forget_df, retain_df


def save_splits(train_df, val_df, test_df, forget_df, retain_df,
                dataset_name, strategy, output_dir="data/processed"):
    """Save all splits to pickle for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"splits_{dataset_name}_{strategy}.pkl")
    with open(path, "wb") as f:
        pickle.dump({
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "forget": forget_df,
            "retain": retain_df,
        }, f)
    print(f"[Data] Splits saved to {path}")
    return path


def load_splits(dataset_name, strategy, output_dir="data/processed"):
    """Load saved splits from pickle."""
    path = os.path.join(output_dir, f"splits_{dataset_name}_{strategy}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────
# High-level preparation (wraps existing prepare_datasets)
# ──────────────────────────────────────────────

def prepare_all(dataset_name="german", forget_strategy="random",
                forget_frac=0.1, seed=42, data_dir="data/raw",
                force_rebuild=False):
    """
    Top-level entry point. Returns the same dict as data.datasets.prepare_datasets
    but also saves splits to data/processed/ for reproducibility.

    Returns dict with keys:
      full_train, val, test, forget, retain -- CreditDataset objects
      forget_indices, retain_indices
      cat_dims, num_num_features
    """
    # Check for cached splits
    splits_path = os.path.join("data/processed",
                               f"splits_{dataset_name}_{forget_strategy}.pkl")
    if os.path.exists(splits_path) and not force_rebuild:
        print(f"[Data] Loading cached splits from {splits_path}")

    # Use the existing prepare_datasets which handles synthetic fallback
    data = prepare_datasets(
        dataset_name=dataset_name,
        data_dir=data_dir,
        forget_strategy=forget_strategy,
        forget_frac=forget_frac,
        seed=seed,
    )

    # Save processed split payload for reuse across methods
    with open(splits_path, "wb") as f:
        pickle.dump(
            {
                "full_train": data["full_train"],
                "val": data["val"],
                "test": data["test"],
                "forget": data["forget"],
                "retain": data["retain"],
                "forget_indices": data["forget_indices"],
                "retain_indices": data["retain_indices"],
                "cat_dims": data["cat_dims"],
                "num_num_features": data["num_num_features"],
                "dataset_name": dataset_name,
                "forget_strategy": forget_strategy,
                "forget_frac": forget_frac,
                "seed": seed,
            },
            f,
        )

    # Save index information for reproducibility
    os.makedirs("data/processed", exist_ok=True)
    index_path = os.path.join("data/processed",
                              f"indices_{dataset_name}_{forget_strategy}.pkl")
    with open(index_path, "wb") as f:
        pickle.dump({
            "forget_indices": data["forget_indices"],
            "retain_indices": data["retain_indices"],
        }, f)

    # Verify forget/retain isolation
    f_set = set(data["forget_indices"].tolist())
    r_set = set(data["retain_indices"].tolist())
    assert len(f_set & r_set) == 0, "FATAL: Forget and retain sets overlap!"
    print(f"[Data] Forget/retain isolation verified: {len(f_set)} forget, {len(r_set)} retain, 0 overlap")

    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare credit scoring datasets")
    parser.add_argument("--dataset", choices=["german", "gmsc"], default="german")
    parser.add_argument("--strategy", choices=["random", "demographic", "temporal"],
                        default="random")
    parser.add_argument("--forget_frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = prepare_all(
        dataset_name=args.dataset,
        forget_strategy=args.strategy,
        forget_frac=args.forget_frac,
        seed=args.seed,
    )

    print(f"\nDataset: {args.dataset}")
    print(f"  Full train: {len(data['full_train'])}")
    print(f"  Validation: {len(data['val'])}")
    print(f"  Test:       {len(data['test'])}")
    print(f"  Forget:     {len(data['forget'])}")
    print(f"  Retain:     {len(data['retain'])}")
    print(f"  Num features: {data['num_num_features']}")
    print(f"  Cat dims:     {data['cat_dims']}")
