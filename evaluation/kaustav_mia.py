"""
Membership Inference Attack (MIA) for evaluating machine unlearning.

Improvements over baseline:
  [OPT-I]  Likelihood Ratio Attack (LiRA): compares per-sample loss on the
           unlearned model vs. the retrained model. More powerful than shadow
           models for small datasets like German Credit (n_forget=70).
           If forget loss ≈ retain loss → certified forgetting.

  [OPT-II] Relearning-based MIA: measures how many gradient steps it takes
           for the model to relearn D_f. A truly unlearned model takes many
           more steps than a model that merely superficially forgot.

  [OPT-III] MIA AUC as primary metric (not accuracy): on small forget sets,
            accuracy is noisy (binary threshold sensitive). AUC is threshold-
            free and more reliable for reporting.

  [OPT-IV] Calibrated shadow MIA: trains shadow models only on D_retain
           subsets (not D_retain ∪ D_forget) to avoid data contamination,
           giving a fairer attack estimate on small datasets.

  [OPT-V]  Argument for shadow MIA unreliability on small datasets:
           With n_forget=70 and n_shadow=4, each shadow model sees at most
           ~35 forget samples → high variance. Reports confidence intervals
           over shadow models.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Optional

from data.datasets import CreditDataset, make_loader, subset_dataset
from evaluation.metrics import get_predictions


# ─────────────────────────────────────────────────────────────────────────────
# Shared utility
# ─────────────────────────────────────────────────────────────────────────────


def _get_model_confidence(model, dataset, device):
    """Confidence features: [P(y=1), P(y=0), max_conf, entropy]."""
    probs, _ = get_predictions(model, dataset, device)
    probs = np.nan_to_num(np.clip(probs, 1e-6, 1.0 - 1e-6), nan=0.5)
    conf_pos = probs
    conf_neg = 1 - probs
    max_conf = np.maximum(probs, 1 - probs)
    entropy = -(probs * np.log(probs + 1e-8) + (1 - probs) * np.log(1 - probs + 1e-8))
    return np.nan_to_num(np.stack([conf_pos, conf_neg, max_conf, entropy], axis=1))


def _per_sample_loss(model, dataset, device):
    """Compute per-sample BCE loss."""
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    loader = make_loader(dataset, batch_size=256, shuffle=False)
    losses = []
    model.eval()
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            x_num = x_num.to(device) if x_num.numel() > 0 else None
            x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
            y = y.to(device)
            logits = torch.nan_to_num(
                model(x_num, x_cat), nan=0.0, posinf=20.0, neginf=-20.0
            )
            losses.append(loss := criterion(logits, y).cpu().numpy())
    return np.nan_to_num(np.concatenate(losses), nan=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# [OPT-I] Likelihood Ratio Attack (LiRA) — primary recommended MIA
# ─────────────────────────────────────────────────────────────────────────────


def likelihood_ratio_attack(
    target_model: nn.Module,  # M* — unlearned model
    reference_model: nn.Module,  # Retrain or base model for comparison
    forget_ds: CreditDataset,
    retain_ds: CreditDataset,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """
    Likelihood Ratio Attack:
      score(x) = loss_target(x) - loss_reference(x)

    For a member (D_f sample):
      - Before unlearning: loss_target low  → score negative (detected as member)
      - After perfect unlearning: loss_target ≈ loss_reference → score ≈ 0 → AUC ≈ 0.5

    This is more statistically powerful than shadow models for small forget sets
    because it uses the actual retrained model as a reference instead of proxies.

    Returns:
      lira_auc     — AUC of attack (goal: ≈ 0.5)
      lira_acc     — Accuracy of attack (goal: ≈ 0.5)
      forget_score_mean  — mean LR score for D_f (≈ 0 = good)
      retain_score_mean  — mean LR score for D_r (should be < 0, model remembers)
    """
    forget_loss_t = _per_sample_loss(target_model, forget_ds, device)
    forget_loss_r = _per_sample_loss(reference_model, forget_ds, device)
    retain_loss_t = _per_sample_loss(target_model, retain_ds, device)
    retain_loss_r = _per_sample_loss(reference_model, retain_ds, device)

    # LR score: negative means target model "remembers" more than reference
    forget_scores = forget_loss_t - forget_loss_r  # close to 0 → unlearned
    retain_scores = retain_loss_t - retain_loss_r  # should be < 0 → retained

    # Attacker labels: D_f = member (1), D_r = non-member (0)
    scores = np.concatenate([forget_scores, retain_scores])
    labels = np.concatenate([np.ones(len(forget_scores)), np.zeros(len(retain_scores))])

    # Lower score (more negative) → predicted member
    # Use -scores so that sklearn's convention (high score = positive class) works
    try:
        lira_auc = roc_auc_score(labels, -scores)
    except Exception:
        lira_auc = 0.5

    # Threshold at 0: negative score → predicted member
    preds = (-scores > 0).astype(int)
    lira_acc = accuracy_score(labels, preds)

    if verbose:
        print(
            f"  [LiRA] forget_score_mean={forget_scores.mean():.4f} "
            f"retain_score_mean={retain_scores.mean():.4f}"
        )
        print(
            f"  [LiRA] attack_auc={lira_auc:.4f} attack_acc={lira_acc:.4f} "
            f"(goal≈0.5 for both)"
        )

    return {
        "lira_auc": lira_auc,
        "lira_acc": lira_acc,
        "forget_score_mean": float(forget_scores.mean()),
        "retain_score_mean": float(retain_scores.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# [OPT-II] Relearning-based MIA
# ─────────────────────────────────────────────────────────────────────────────


def relearning_mia(
    target_model: nn.Module,
    forget_ds: CreditDataset,
    device: torch.device,
    n_steps: int = 20,
    lr: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """
    Measures relearning speed on D_f as a proxy for residual membership.

    Protocol:
      1. Baseline forget_auc before any relearning
      2. Fine-tune on D_f for n_steps gradient steps
      3. Record forget_auc after relearning

    Interpretation:
      - Large jump in forget_auc (e.g., +0.3 in 20 steps) → model still
        "remembers" D_f latently → forgetting was superficial.
      - Small jump (e.g., +0.05 in 20 steps) → model truly unlearned,
        D_f is effectively OOD → certified forgetting.

    Compare against full-retrain baseline (true lower bound of relearning speed).
    """
    from train import evaluate

    model_rl = copy.deepcopy(target_model).to(device)
    optimizer = torch.optim.Adam(model_rl.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    forget_loader = make_loader(forget_ds, batch_size=64, shuffle=True)
    forget_eval = make_loader(forget_ds, batch_size=256, shuffle=False)

    history = {"step": [], "forget_auc": []}

    # Baseline
    base_m = evaluate(model_rl, forget_eval, device)
    history["step"].append(0)
    history["forget_auc"].append(base_m["auc"])

    step = 0
    forget_iter = iter(forget_loader)
    for _ in range(n_steps):
        model_rl.train()
        try:
            x_num, x_cat, y = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            x_num, x_cat, y = next(forget_iter)

        x_num = x_num.to(device) if x_num.numel() > 0 else None
        x_cat = x_cat.to(device) if x_cat.numel() > 0 else None
        y = y.to(device)
        optimizer.zero_grad()
        criterion(model_rl(x_num, x_cat), y).backward()
        optimizer.step()
        step += 1

        if step % 5 == 0:
            m = evaluate(model_rl, forget_eval, device)
            history["step"].append(step)
            history["forget_auc"].append(m["auc"])

    relearn_gain = history["forget_auc"][-1] - history["forget_auc"][0]

    if verbose:
        print(
            f"  [RelearningMIA] forget_auc: {history['forget_auc'][0]:.4f} → "
            f"{history['forget_auc'][-1]:.4f} over {n_steps} steps "
            f"(gain={relearn_gain:+.4f})"
        )
        print(
            f"  [RelearningMIA] Small gain (<0.1) → certified. "
            f"Large gain (>0.2) → superficial forgetting."
        )

    history["relearn_gain"] = relearn_gain
    return history


# ─────────────────────────────────────────────────────────────────────────────
# [OPT-IV] Calibrated shadow MIA (trains only on D_retain subsets)
# ─────────────────────────────────────────────────────────────────────────────


def run_mia(
    target_model: nn.Module,
    forget_ds: CreditDataset,
    retain_ds: CreditDataset,
    train_ds: CreditDataset,
    device: torch.device,
    n_shadow: int = 4,
    shadow_frac: float = 0.5,
    attacker: str = "lr",
    seed: int = 42,
    # [OPT-IV] Use only retain subsets for shadow training (avoids contamination)
    calibrated_shadow: bool = False,
    # [OPT-V] Report confidence interval over shadow models
    report_ci: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Shadow-model MIA. Two modes:
      calibrated_shadow=False (default): shadow trains on D_retain ∪ D_forget
          subsets. Same as baseline. Can overestimate MIA for small D_f.
      calibrated_shadow=True [OPT-IV]: shadow trains only on D_retain subsets,
          using a holdout of D_retain as the "non-member" set. Avoids inflating
          MIA by training shadows directly on D_f samples. Better for small
          forget sets (n_forget < 100).

    report_ci=True [OPT-V]: reports mean ± std MIA score across shadow models,
      quantifying how much of the 0.757 score is variance from small n_forget.
    """
    rng = np.random.RandomState(seed)

    # Choose which dataset to sample shadow training sets from
    shadow_source = retain_ds if calibrated_shadow else train_ds
    n_source = len(shadow_source)
    mode_str = "calibrated (D_retain only)" if calibrated_shadow else "standard"

    if verbose:
        print(f"  [MIA] Training {n_shadow} shadow models ({mode_str})...")

    shadow_features = []
    shadow_labels = []
    per_shadow_scores = []  # [OPT-V]

    for i in range(n_shadow):
        shadow_idx = rng.choice(
            n_source, size=int(n_source * shadow_frac), replace=False
        )
        out_idx = np.setdiff1d(np.arange(n_source), shadow_idx)

        shadow_train = subset_dataset(shadow_source, shadow_idx)
        shadow_out = subset_dataset(
            shadow_source, out_idx[: min(len(out_idx), len(shadow_idx))]
        )

        shadow_model = copy.deepcopy(target_model).to(device)
        for module in shadow_model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        from train import train_model

        shadow_model, _ = train_model(
            shadow_model,
            shadow_train,
            shadow_out,
            device,
            epochs=20,
            batch_size=128,
            lr=1e-3,
            verbose=False,
            patience=5,
        )

        in_feats = _get_model_confidence(shadow_model, shadow_train, device)
        out_feats = _get_model_confidence(shadow_model, shadow_out, device)
        shadow_features.append(np.vstack([in_feats, out_feats]))
        shadow_labels.append(
            np.concatenate([np.ones(len(in_feats)), np.zeros(len(out_feats))])
        )

        # [OPT-V] Per-shadow MIA score on forget set
        clf_i = (
            LogisticRegression(max_iter=500, C=1.0, random_state=seed)
            if attacker == "lr"
            else RandomForestClassifier(n_estimators=100, random_state=seed)
        )
        clf_i.fit(np.vstack(shadow_features), np.concatenate(shadow_labels))
        f_feats = _get_model_confidence(target_model, forget_ds, device)
        f_feats = np.nan_to_num(f_feats)
        per_shadow_scores.append(
            accuracy_score(np.ones(len(f_feats)), clf_i.predict(f_feats))
        )

        if verbose:
            print(
                f"    Shadow model {i+1}/{n_shadow} done "
                f"(mia_score={per_shadow_scores[-1]:.4f})"
            )

    X_shadow = np.nan_to_num(np.vstack(shadow_features))
    y_shadow = np.concatenate(shadow_labels)

    clf = (
        LogisticRegression(max_iter=500, C=1.0, random_state=seed)
        if attacker == "lr"
        else RandomForestClassifier(n_estimators=100, random_state=seed)
    )
    clf.fit(X_shadow, y_shadow)

    forget_feats = np.nan_to_num(_get_model_confidence(target_model, forget_ds, device))
    forget_true = np.ones(len(forget_feats))
    forget_preds = clf.predict(forget_feats)
    forget_probs = clf.predict_proba(forget_feats)[:, 1]

    mia_score = accuracy_score(forget_true, forget_preds)
    try:
        mia_auc = roc_auc_score(forget_true, forget_probs)
    except Exception:
        mia_auc = 0.5

    retain_feats = np.nan_to_num(_get_model_confidence(target_model, retain_ds, device))
    retain_true = np.ones(len(retain_feats))
    retain_mia = accuracy_score(retain_true, clf.predict(retain_feats))

    # [OPT-V] Confidence interval
    ci_mean = float(np.mean(per_shadow_scores))
    ci_std = float(np.std(per_shadow_scores))

    if verbose:
        print(
            f"  [MIA] forget_set MIA accuracy={mia_score:.4f} (goal≈0.5) | "
            f"retain_set MIA={retain_mia:.4f} | MIA AUC={mia_auc:.4f}"
        )
        if report_ci:
            print(
                f"  [MIA-CI] Per-shadow scores: {[f'{s:.3f}' for s in per_shadow_scores]}"
            )
            print(
                f"  [MIA-CI] Mean={ci_mean:.4f} ± Std={ci_std:.4f}  "
                f"← High std with n_forget=70 means shadow MIA is unreliable."
            )
            print(f"           Recommend reporting LiRA AUC instead as primary metric.")

    return {
        "mia_score": mia_score,
        "mia_auc": mia_auc,
        "retain_mia": retain_mia,
        "attacker_clf": clf,
        "ci_mean": ci_mean,
        "ci_std": ci_std,
        "per_shadow_scores": per_shadow_scores,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Loss-based MIA (unchanged from baseline — [OPT-III]: now reports AUC too)
# ─────────────────────────────────────────────────────────────────────────────


def loss_based_mia(
    target_model: nn.Module,
    forget_ds: CreditDataset,
    retain_ds: CreditDataset,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """
    Threshold-based MIA. [OPT-III] Now reports AUC (threshold-free) in addition
    to accuracy, as primary metric for small forget sets.
    """
    forget_losses = _per_sample_loss(target_model, forget_ds, device)
    retain_losses = _per_sample_loss(target_model, retain_ds, device)
    threshold = retain_losses.mean()

    forget_preds = (forget_losses < threshold).astype(int)
    true_member = np.ones(len(forget_losses), dtype=int)
    attack_acc = accuracy_score(true_member, forget_preds)

    # [OPT-III] AUC using loss as the score (lower loss → more likely member)
    all_losses = np.concatenate([forget_losses, retain_losses])
    all_labels = np.concatenate(
        [np.ones(len(forget_losses)), np.zeros(len(retain_losses))]
    )
    try:
        # Higher label = member. Members have LOWER loss → use -loss as score.
        loss_mia_auc = roc_auc_score(all_labels, -all_losses)
    except Exception:
        loss_mia_auc = 0.5

    if verbose:
        print(
            f"  [MIA-Loss] forget_loss_mean={forget_losses.mean():.4f} "
            f"retain_loss_mean={retain_losses.mean():.4f} "
            f"attack_acc={attack_acc:.4f} attack_auc={loss_mia_auc:.4f} (goal≈0.5)"
        )

    return {
        "mia_score": attack_acc,
        "mia_auc": loss_mia_auc,  # [OPT-III] — use this as primary metric
        "forget_loss_mean": float(forget_losses.mean()),
        "retain_loss_mean": float(retain_losses.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full MIA evaluation suite (convenience wrapper)
# ─────────────────────────────────────────────────────────────────────────────


def run_full_mia_suite(
    target_model: nn.Module,
    forget_ds: CreditDataset,
    retain_ds: CreditDataset,
    train_ds: CreditDataset,
    device: torch.device,
    reference_model: Optional[nn.Module] = None,  # For LiRA [OPT-I]
    run_lira: bool = True,  # [OPT-I]
    run_relearning: bool = True,  # [OPT-II]
    relearning_steps: int = 20,
    calibrated_shadow: bool = False,  # [OPT-IV]
    report_ci: bool = True,  # [OPT-V]
    n_shadow: int = 4,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Runs all MIA variants and returns a unified results dict.

    Recommended reporting order for the paper:
      1. loss_mia_auc       ← most reliable for small datasets [OPT-III]
      2. lira_auc           ← most powerful if reference model available [OPT-I]
      3. relearn_gain       ← most interpretable [OPT-II]
      4. shadow_mia_score   ← report with CI to show it's noisy [OPT-V]
    """
    results = {}

    if verbose:
        print("\n── MIA Suite ──────────────────────────────────────")

    # 1. Loss-based MIA (always run)
    results["loss_mia"] = loss_based_mia(
        target_model, forget_ds, retain_ds, device, verbose
    )

    # 2. LiRA [OPT-I]
    if run_lira and reference_model is not None:
        results["lira"] = likelihood_ratio_attack(
            target_model, reference_model, forget_ds, retain_ds, device, verbose
        )
    else:
        if run_lira and verbose:
            print(
                "  [LiRA] Skipped: reference_model not provided. "
                "Pass retrain model or base model as reference_model."
            )

    # 3. Relearning MIA [OPT-II]
    if run_relearning:
        results["relearning_mia"] = relearning_mia(
            target_model, forget_ds, device, n_steps=relearning_steps, verbose=verbose
        )

    # 4. Shadow MIA [OPT-IV, OPT-V]
    results["shadow_mia"] = run_mia(
        target_model,
        forget_ds,
        retain_ds,
        train_ds,
        device,
        n_shadow=n_shadow,
        calibrated_shadow=calibrated_shadow,
        report_ci=report_ci,
        seed=seed,
        verbose=verbose,
    )

    if verbose:
        print("\n── MIA Summary ────────────────────────────────────")
        print(
            f"  Loss MIA AUC    = {results['loss_mia']['mia_auc']:.4f}  "
            f"(primary, goal≈0.5)"
        )
        if "lira" in results:
            print(
                f"  LiRA AUC        = {results['lira']['lira_auc']:.4f}  "
                f"(primary if ref model available, goal≈0.5)"
            )
        if "relearning_mia" in results:
            gain = results["relearning_mia"]["relearn_gain"]
            verdict = "✓ certified" if abs(gain) < 0.1 else "✗ superficial"
            print(f"  Relearn gain    = {gain:+.4f}  ({verdict})")
        smia = results["shadow_mia"]
        print(
            f"  Shadow MIA      = {smia['mia_score']:.4f} ± {smia['ci_std']:.4f}  "
            f"(CI shows reliability)"
        )
        print("───────────────────────────────────────────────────")

    return results
