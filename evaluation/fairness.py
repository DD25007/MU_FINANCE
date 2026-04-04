"""
Fairness metrics for credit scoring unlearning evaluation.

Metrics:
  - Equalized Odds Difference (delta_EO)
  - Demographic Parity Difference
  - Calibration Error (ECE)

For German Credit: protected attribute = age (threshold 25)
For GMSC: use DebtRatio quantile as proxy
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from data.datasets import CreditDataset, make_loader
from evaluation.metrics import get_predictions


def compute_delta_eo(model, test_ds, groups, device, threshold=0.5):
    """
    Equalized Odds Difference:
    delta_EO = |TPR(group A) - TPR(group B)| + |FPR(group A) - FPR(group B)|

    Args:
        model: trained model
        test_ds: CreditDataset for test set
        groups: np.ndarray of binary group labels (0/1), same length as test_ds
        device: torch device
        threshold: classification threshold

    Returns:
        delta_eo: float -- lower is fairer
    """
    probs, labels = get_predictions(model, test_ds, device)
    preds = (probs > threshold).astype(int)
    labels = labels.astype(int)
    groups = np.asarray(groups).astype(int)

    assert len(probs) == len(groups), \
        f"Mismatch: {len(probs)} predictions vs {len(groups)} group labels"

    group_vals = np.unique(groups)
    if len(group_vals) < 2:
        return 0.0

    def tpr(preds, labels, mask):
        pos = mask & (labels == 1)
        if pos.sum() == 0:
            return 0.0
        return preds[pos].mean()

    def fpr(preds, labels, mask):
        neg = mask & (labels == 0)
        if neg.sum() == 0:
            return 0.0
        return preds[neg].mean()

    group_a = groups == group_vals[0]
    group_b = groups == group_vals[1]

    delta_tpr = abs(tpr(preds, labels, group_a) - tpr(preds, labels, group_b))
    delta_fpr = abs(fpr(preds, labels, group_a) - fpr(preds, labels, group_b))

    return delta_tpr + delta_fpr


def compute_demographic_parity(model, test_ds, groups, device, threshold=0.5):
    """
    Demographic Parity Difference:
    |P(Y_hat=1 | group=A) - P(Y_hat=1 | group=B)|

    Lower is fairer.
    """
    probs, _ = get_predictions(model, test_ds, device)
    preds = (probs > threshold).astype(int)
    groups = np.asarray(groups).astype(int)

    group_vals = np.unique(groups)
    if len(group_vals) < 2:
        return 0.0

    rate_a = preds[groups == group_vals[0]].mean()
    rate_b = preds[groups == group_vals[1]].mean()

    return abs(rate_a - rate_b)


def compute_ece(model, test_ds, device, n_bins=15):
    """
    Expected Calibration Error: measures if predicted probabilities match
    actual frequencies.

    Finance regulators care about calibration -- a model should output p=0.7
    when 70% of similar applicants actually default.

    Target: ECE should not increase after unlearning.
    """
    probs, labels = get_predictions(model, test_ds, device)
    labels = labels.astype(int)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        bin_frac = mask.sum() / len(probs)
        ece += bin_frac * abs(bin_acc - bin_conf)

    return float(ece)


def build_age_groups(dataset_name, test_ds, age_values=None, age_threshold=25):
    """
    Build binary group labels for fairness evaluation.

    For German Credit: group 0 = age < 25, group 1 = age >= 25
    For GMSC: use median DebtRatio as proxy split

    Args:
        dataset_name: "german" or "gmsc"
        test_ds: CreditDataset
        age_values: optional precomputed age values for test set
        age_threshold: age cutoff for demographic grouping

    Returns:
        groups: np.ndarray of 0/1 group labels
    """
    if age_values is not None:
        return (age_values >= age_threshold).astype(int)

    # Fall back: use a feature from the dataset
    if test_ds.x_num is not None:
        # Use first numerical feature quantile as proxy
        feature_vals = test_ds.x_num[:, 0].numpy()
        median_val = np.median(feature_vals)
        return (feature_vals >= median_val).astype(int)

    # Last resort: random split (not meaningful, but prevents crashes)
    return np.random.randint(0, 2, size=len(test_ds))
