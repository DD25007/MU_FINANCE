"""
Dataset loading, preprocessing, and forget-set construction for credit scoring.

Supported datasets:
  - German Credit (UCI)          — 1,000 samples, 20 features
  - Give Me Some Credit (Kaggle) — 150,000 samples, 10 features
  - LendingClub (temporal)       — 2M+ records, temporal splits

Forget-set strategies:
  1. Random — random 5/10/20% of training data
  2. Demographic — age-based subgroup (age < 25 for German Credit)
  3. Temporal — loans from 2008-2009 (LendingClub)
"""

import os
import io
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader, Subset


# ──────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────

@dataclass
class CreditDataset(Dataset):
    """Holds preprocessed numeric + categorical tensors and labels."""
    x_num: Optional[torch.Tensor]  # (N, num_num_features)
    x_cat: Optional[torch.Tensor]  # (N, num_cat_features) — integer indices
    y: torch.Tensor                # (N,) — float32 binary labels
    cat_dims: List[int] = field(default_factory=list)  # cardinalities per cat feature
    num_num_features: int = 0
    feature_names: List[str] = field(default_factory=list)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_num = self.x_num[idx] if self.x_num is not None else torch.tensor([])
        x_cat = self.x_cat[idx] if self.x_cat is not None else torch.tensor([], dtype=torch.long)
        return x_num, x_cat, self.y[idx]


def make_loader(dataset: CreditDataset, batch_size: int = 256, shuffle: bool = True,
                num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def subset_dataset(dataset: CreditDataset, indices: np.ndarray) -> CreditDataset:
    x_num = dataset.x_num[indices] if dataset.x_num is not None else None
    x_cat = dataset.x_cat[indices] if dataset.x_cat is not None else None
    return CreditDataset(
        x_num=x_num, x_cat=x_cat, y=dataset.y[indices],
        cat_dims=dataset.cat_dims,
        num_num_features=dataset.num_num_features,
        feature_names=dataset.feature_names,
    )


# ──────────────────────────────────────────────
# German Credit (UCI) — primary dataset
# ──────────────────────────────────────────────

def load_german_credit(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load German Credit from local file or generate synthetic proxy."""
    path = os.path.join(data_dir, "german.data")
    if os.path.exists(path):
        cols = [
            "status", "duration", "credit_history", "purpose", "amount",
            "savings", "employment", "installment_rate", "personal_status_sex",
            "other_debtors", "residence_since", "property", "age",
            "other_installment", "housing", "existing_credits", "job",
            "liable_people", "telephone", "foreign_worker", "target"
        ]
        df = pd.read_csv(path, sep=" ", header=None, names=cols)
        df["target"] = (df["target"] == 2).astype(int)  # 2=bad → 1, 1=good → 0
    else:
        print("[Data] german.data not found — generating synthetic proxy (1000 samples)")
        df = _synthetic_german_credit(1000)
    return df


def _synthetic_german_credit(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Synthetic German Credit with same schema for offline testing."""
    rng = np.random.RandomState(seed)
    cat_cols = {
        "status": ["A11", "A12", "A13", "A14"],
        "credit_history": ["A30", "A31", "A32", "A33", "A34"],
        "purpose": ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49"],
        "savings": ["A61", "A62", "A63", "A64", "A65"],
        "employment": ["A71", "A72", "A73", "A74", "A75"],
        "personal_status_sex": ["A91", "A92", "A93", "A94"],
        "other_debtors": ["A101", "A102", "A103"],
        "property": ["A121", "A122", "A123", "A124"],
        "other_installment": ["A141", "A142", "A143"],
        "housing": ["A151", "A152", "A153"],
        "job": ["A171", "A172", "A173", "A174"],
        "telephone": ["A191", "A192"],
        "foreign_worker": ["A201", "A202"],
    }
    num_cols = {
        "duration": (4, 72),
        "amount": (250, 18424),
        "installment_rate": (1, 4),
        "residence_since": (1, 4),
        "age": (19, 75),
        "existing_credits": (1, 4),
        "liable_people": (1, 2),
    }
    data = {}
    for col, cats in cat_cols.items():
        data[col] = rng.choice(cats, n)
    for col, (lo, hi) in num_cols.items():
        data[col] = rng.randint(lo, hi + 1, n)
    # Synthetic label: higher amount + shorter duration → higher default risk
    logit = (data["amount"] - 4000) / 3000 - (data["duration"] - 20) / 15
    prob = 1 / (1 + np.exp(-logit + rng.randn(n) * 0.5))
    data["target"] = (prob > 0.5).astype(int)
    return pd.DataFrame(data)


def preprocess_german_credit(df: pd.DataFrame):
    """Preprocess German Credit: encode, normalise, return tensors + metadata."""
    cat_feature_names = [
        "status", "credit_history", "purpose", "savings", "employment",
        "personal_status_sex", "other_debtors", "property",
        "other_installment", "housing", "job", "telephone", "foreign_worker"
    ]
    num_feature_names = [
        "duration", "amount", "installment_rate", "residence_since",
        "age", "existing_credits", "liable_people"
    ]

    # Encode categoricals to integers
    cat_dims = []
    cat_arrays = []
    for col in cat_feature_names:
        cats = df[col].astype("category")
        cat_arrays.append(cats.cat.codes.values)
        cat_dims.append(int(cats.nunique()))

    # Normalise numerics
    num_arrays = []
    for col in num_feature_names:
        vals = df[col].astype(float).values
        vals = (vals - vals.mean()) / (vals.std() + 1e-8)
        num_arrays.append(vals)

    x_num = torch.tensor(np.stack(num_arrays, axis=1), dtype=torch.float32)
    x_cat = torch.tensor(np.stack(cat_arrays, axis=1), dtype=torch.long)
    y = torch.tensor(df["target"].values, dtype=torch.float32)

    return CreditDataset(
        x_num=x_num, x_cat=x_cat, y=y,
        cat_dims=cat_dims,
        num_num_features=len(num_feature_names),
        feature_names=num_feature_names + cat_feature_names,
    ), df["age"].values  # return age for demographic forget-set


# ──────────────────────────────────────────────
# Give Me Some Credit (Kaggle)
# ──────────────────────────────────────────────

def load_give_me_some_credit(data_dir: str = "data/raw") -> pd.DataFrame:
    path = os.path.join(data_dir, "cs-training.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        df = df.rename(columns={"SeriousDlqin2yrs": "target"})
        # Coerce unparseable values (e.g. European thousand-separator "1.046.279")
        # to NaN, then drop those rows
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        df = df.reset_index(drop=True)
    else:
        print("[Data] cs-training.csv not found — generating synthetic proxy (5000 samples)")
        df = _synthetic_gmsc(5000)
    return df


def _synthetic_gmsc(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    data = {
        "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 1.5, n),
        "age": rng.randint(21, 90, n),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.poisson(0.2, n),
        "DebtRatio": rng.uniform(0, 5, n),
        "MonthlyIncome": rng.exponential(5000, n),
        "NumberOfOpenCreditLinesAndLoans": rng.randint(0, 20, n),
        "NumberOfTimes90DaysLate": rng.poisson(0.1, n),
        "NumberRealEstateLoansOrLines": rng.randint(0, 5, n),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.poisson(0.1, n),
        "NumberOfDependents": rng.randint(0, 5, n),
    }
    logit = (data["DebtRatio"] - 2) + 0.5 * data["NumberOfTimes90DaysLate"] + rng.randn(n) * 0.3
    prob = 1 / (1 + np.exp(-logit))
    data["target"] = (prob > 0.5).astype(int)
    return pd.DataFrame(data)


def preprocess_gmsc(df: pd.DataFrame):
    num_feature_names = [
        "RevolvingUtilizationOfUnsecuredLines", "age",
        "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
        "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"
    ]
    # Clip extreme values and normalise
    num_arrays = []
    for col in num_feature_names:
        vals = df[col].astype(float).values
        p1, p99 = np.percentile(vals, 1), np.percentile(vals, 99)
        vals = np.clip(vals, p1, p99)
        vals = (vals - vals.mean()) / (vals.std() + 1e-8)
        num_arrays.append(vals)

    x_num = torch.tensor(np.stack(num_arrays, axis=1), dtype=torch.float32)
    y = torch.tensor(df["target"].values, dtype=torch.float32)

    return CreditDataset(
        x_num=x_num, x_cat=None, y=y,
        cat_dims=[],
        num_num_features=len(num_feature_names),
        feature_names=num_feature_names,
    ), df["age"].values


# ──────────────────────────────────────────────
# Forget-set construction
# ──────────────────────────────────────────────

def make_forget_set_random(n: int, frac: float = 0.1, seed: int = 42) -> np.ndarray:
    """Random forget-set: frac of training indices."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=int(n * frac), replace=False)
    return idx


def make_forget_set_demographic(age_values: np.ndarray, age_threshold: int = 25) -> np.ndarray:
    """Forget all samples where age < threshold (young demographic)."""
    return np.where(age_values < age_threshold)[0]


def make_forget_set_temporal(year_values: np.ndarray, start: int = 2007, end: int = 2009) -> np.ndarray:
    """Forget samples from a specific time window (LendingClub)."""
    return np.where((year_values >= start) & (year_values <= end))[0]


# ──────────────────────────────────────────────
# Full dataset preparation
# ──────────────────────────────────────────────

def prepare_datasets(
    dataset_name: str = "german",
    data_dir: str = "data/raw",
    forget_strategy: str = "random",
    forget_frac: float = 0.1,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Returns a dict with keys:
      full_train, val, test  — CreditDataset splits
      forget, retain         — forget and retain subsets of full_train
      cat_dims, num_num_features
      forget_indices, retain_indices  — integer arrays into full_train
    """
    rng = np.random.RandomState(seed)

    if dataset_name == "german":
        df = load_german_credit(data_dir)
        dataset, age_vals = preprocess_german_credit(df)
    elif dataset_name == "gmsc":
        df = load_give_me_some_credit(data_dir)
        dataset, age_vals = preprocess_gmsc(df)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    N = len(dataset)
    indices = rng.permutation(N)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    full_train = subset_dataset(dataset, train_idx)
    val_ds = subset_dataset(dataset, val_idx)
    test_ds = subset_dataset(dataset, test_idx)

    # Build forget set indices (relative to full_train)
    train_age = age_vals[train_idx] if age_vals is not None else None

    if forget_strategy == "random":
        f_idx = make_forget_set_random(len(full_train), forget_frac, seed)
    elif forget_strategy == "demographic" and train_age is not None:
        f_idx = make_forget_set_demographic(train_age, age_threshold=25)
        if len(f_idx) == 0:
            print("[Warn] No samples with age<25; falling back to random 10%")
            f_idx = make_forget_set_random(len(full_train), 0.10, seed)
    else:
        f_idx = make_forget_set_random(len(full_train), forget_frac, seed)

    all_idx = np.arange(len(full_train))
    r_idx = np.setdiff1d(all_idx, f_idx)

    forget_ds = subset_dataset(full_train, f_idx)
    retain_ds = subset_dataset(full_train, r_idx)

    print(f"[Data] {dataset_name} | train={len(full_train)} val={len(val_ds)} test={len(test_ds)}")
    print(f"[Data] forget={len(forget_ds)} ({forget_strategy}) | retain={len(retain_ds)}")

    return {
        "full_train": full_train,
        "val": val_ds,
        "test": test_ds,
        "forget": forget_ds,
        "retain": retain_ds,
        "forget_indices": f_idx,
        "retain_indices": r_idx,
        "cat_dims": dataset.cat_dims,
        "num_num_features": dataset.num_num_features,
    }
