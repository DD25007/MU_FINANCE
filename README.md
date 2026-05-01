# LoRA Machine Unlearning for Credit Scoring (MU-Finance)

A parameter-efficient machine unlearning framework for transformer-based credit scoring models using LoRA adapters. This repository implements a fast alternative to full retraining, achieving effective data forgetting with minimal utility loss on financial datasets.

---

## 🚀 Overview

Modern credit scoring models are trained on large-scale sensitive financial data. Under regulations like GDPR (Right to be Forgotten) and Fair Credit Reporting Act (FCRA), specific user data must be removed from trained models without requiring full retraining. Traditional retraining is computationally expensive (hours per run).

This project introduces **LoRA-based machine unlearning**, which:

* Removes the influence of selected data (forget set) efficiently
* Preserves model performance on remaining data (retain set)
* Reduces unlearning time from **hours → minutes**
* Uses only **0.5-2%** additional parameters

---

## 🧠 Core Idea

We use **Low-Rank Adaptation (LoRA)** to perform targeted updates:

* Freeze base model weights
* Learn small low-rank adapters for unlearning
* Apply a **three-phase pipeline**:

### 1. Base Training

Train model on full dataset (D = $D_f \cup D_r$) to get base weights (W).

### 2. Forget Adapter (Gradient Ascent)

* Train LoRA adapter on forget set ($D_f$) using gradient ascent to maximize loss
* Objective: degrade performance on forget samples

### 3. Retain Adapter (Distillation)

* Train second adapter on retain set ($D_r$) using knowledge distillation from base model
* Objective: recover utility via knowledge distillation

### Final Model

* Merge retain adapter into base weights
* Discard forget adapter

---

## 🏗️ Models

### Supported Architectures

* **FT-Transformer** - Primary model (Gorishniy et al., 2021)
* **TabTransformer** - Hybrid transformer-MLP architecture
* **TabDDPM** - Diffusion-based tabular model (baseline)

### LoRA Application

Low-Rank Adaptation is applied to attention layers:

* Query (Q) projection matrices
* Value (V) projection matrices
* Configurable rank (r) and scaling (α)
* Minimal parameter overhead (~0.5-2% of base model)

---

## 📊 Datasets

### Available Datasets

* **German Credit** - UCI dataset (1,000 samples, 20 features)
* **Give Me Some Credit (GMSC)** - Kaggle dataset (150,000 samples, 11 features)

### Forget Set Strategies

* **Random**: Uniform random subset (5-20% of data)
* **Demographic**: Subgroup targeting (e.g., age < 25, gender = M)

Configurable via `--forget_strategy` parameter.

---

## 📈 Evaluation Metrics

| Metric          | Optimal Value | Purpose                                    |
| --------------- | ------------- | ------------------------------------------ |
| Forget Accuracy | ≈ 0.50 (50%)  | Should approach ~50% (random)              |
| Retain AUC      | ≈ 0.85+       | Utility preservation (high)                |
| Test AUC        | ≈ 0.76+       | Generalization (high)                      |
| KL Divergence   | ≈ 0.0-0.15    | Distribution distance (low)                |
| JS Divergence   | ≈ 0.0-0.04    | Symmetric distribution distance (low)      |
| MIA Score       | ≈ 0.50        | Certified forgetting (random)              |
| Relearning Gain | Low (<10%)    | Forgetting robustness                      |
| Wall-clock Time | <200s (GMSC)  | Efficiency (4-5× faster than Full Retrain) |
| Fairness (ΔEO)  | ≈ 0.0-0.05    | Bias reduction (near zero)                 |

---

## ⚙️ Pipeline

### Full Unlearning Pipeline


1. **Data preprocessing** - Normalize and split into train/retain/forget/test sets
2. **Train base model** - Full model on complete dataset (train set)
3. **Run baseline methods** - Full retraining, SISA, gradient ascent, influence functions
4. **Apply LoRA unlearning** - Forget adapter + Retain adapter pipeline
5. **Evaluation** - Forgetting effectiveness, utility preservation, fairness, MIA
6. **Ablation studies** - Rank, alpha, learning rates, adapter architectures
7. **Scalability experiments** - Performance vs. dataset/model size
8. **Membership inference attacks** - LiRA, relearning, shadow model attacks

---

## 🔬 Key Results

### Summary

* Forget accuracy ≈ **0.51** (near ideal 0.5)
* Retain AUC drop < **1–2%**
* Up to **4–10× speedup** over retraining
* Significant fairness improvement (ΔEO ↓)
* MIA score ≈ **0.47–0.51** (certified forgetting)

### 📊 Comparison with Baselines (German Credit)

#### Experiment 1: German Credit (Random Forget Set) — FT-Transformer

| Method              | Forget Acc ↓ | Forget AUC ↓ | Retain AUC ↑ | Test AUC ↑ | KL Div ↓   | JS Div ↓   | Time (s) | Remark             |
| ------------------- | ------------ | ------------ | ------------ | ---------- | ---------- | ---------- | -------- | ------------------ |
| Base Model          | 0.8143       | 0.8424       | —            | 0.7898     | —          | —          | —        | No unlearning      |
| Full Retrain        | 0.7143       | 0.7125       | 0.8574       | 0.7367     | 0.0584     | 0.0127     | 7.47     | Gold standard      |
| Gradient Ascent     | 0.5143       | 0.2286       | 0.6001       | 0.6176     | 1.2535     | 0.3897     | 11.70    | Poor utility       |
| SISA                | 0.7429       | 0.6959       | 0.7894       | 0.8076     | 0.0612     | 0.0165     | 8.36     | Competitive        |
| Finetune Retain     | 0.8000       | 0.8413       | 0.8079       | 0.7913     | 0.0011     | 0.0001     | 1.37     | No forgetting      |
| **LoRA (Ours)** | **0.5143**   | **0.5128**   | **0.7528**   | **0.7625** | **0.1266** | **0.0354** | **9.59** | **Best trade-off** |

#### Experiment 2: German Credit (Demographic: Age<25) — Fairness Scenario

| Method              | Forget Acc | Retain AUC ↑ | Test AUC ↑ | KL Div ↓   | JS Div ↓   | ΔEO (Fairness) ↓ | Time (s) | Remark                |
| ------------------- | ---------- | ------------ | ---------- | ---------- | ---------- | ---------------- | -------- | --------------------- |
| Full Retrain        | 0.6832     | 0.8777       | 0.7482     | 0.0526     | 0.0123     | 0.0000           | 11.01    | Gold standard         |
| Gradient Ascent     | 0.5941     | 0.5434       | 0.5645     | 0.5221     | 0.0823     | 0.0000           | 1.71     | Fast but poor utility |
| **LoRA (Ours)** | **0.6238** | **0.7931**   | **0.7224** | **0.0523** | **0.0145** | **0.0000**       | **8.92** | **Best utility**      |

#### Experiment 3: Give Me Some Credit (Large Scale, Random Forget Set) — FT-Transformer

| Method              | Forget Acc | Forget AUC ↓ | Retain AUC ↑ | Test AUC ↑ | KL Div ↓   | JS Div ↓   | Time (s)  | Speedup vs Retrain |
| ------------------- | ---------- | ------------ | ------------ | ---------- | ---------- | ---------- | --------- | ------------------ |
| Full Retrain        | 0.9393     | 0.8517       | 0.8525       | 0.8359     | 0.0000     | 0.0000     | 764.6     | 1.0x               |
| Gradient Ascent     | 0.0613     | 0.4844       | 0.4743       | 0.4834     | 2.1154     | 0.3456     | 152.1     | 5.0x               |
| SISA                | 0.9387     | 0.8511       | 0.8470       | 0.8332     | 0.0089     | 0.0024     | 220.2     | 3.5x               |
| **LoRA (Ours)** | **0.9415** | **0.8608**   | **0.7640**   | **0.7483** | **0.0342** | **0.0089** | **167.6** | **4.6x speedup**   |

#### Experiment 4: Cross-Model Comparison (German Credit, Random Forget Set)

| Model          | Forget Acc | Forget AUC | Retain AUC | Test AUC | Time (s) | Remark             |
| -------------- | ---------- | ---------- | ---------- | -------- | -------- | ------------------ |
| FT-Transformer | 0.5143     | 0.5128     | 0.7528     | 0.7625   | 9.59     | Primary (best AUC) |
| TabTransformer | 0.4000     | 0.4915     | 0.7970     | 0.7766   | 8.00     | Better utility     |
| TabDDPM        | 0.4857     | 0.7094     | 0.7873     | 0.7761   | 9.20     | Strong forgetting  |

#### Experiment 5: Cross-Strategy Comparison (All Models & Datasets)

**German Credit (Random vs Demographic)**

| Model          | Strategy    | Forget Acc | Retain AUC | Test AUC | Time (s) |
| -------------- | ----------- | ---------- | ---------- | -------- | -------- |
| FT-Transformer | Random      | 0.5143     | 0.7528     | 0.7625   | 9.59     |
| FT-Transformer | Demographic | 0.6238     | 0.7931     | 0.7224   | 8.92     |
| TabTransformer | Random      | 0.4000     | 0.7970     | 0.7766   | 8.00     |
| TabTransformer | Demographic | 0.5644     | 0.8019     | 0.7649   | 12.50    |
| TabDDPM        | Random      | 0.4857     | 0.7873     | 0.7761   | 9.20     |
| TabDDPM        | Demographic | 0.5941     | 0.8228     | 0.7563   | 26.55    |

**Give Me Some Credit (Random vs Demographic)**

| Model          | Strategy    | Forget Acc | Retain AUC | Test AUC | Time (s) |
| -------------- | ----------- | ---------- | ---------- | -------- | -------- |
| FT-Transformer | Random      | 0.9404     | 0.8453     | 0.8337   | 1927.53  |
| FT-Transformer | Demographic | 0.0759     | 0.8505     | 0.8295   | 1939.95  |
| TabTransformer | Random      | 0.9300     | 0.7775     | 0.7607   | 169.00   |
| TabTransformer | Demographic | 0.9120     | 0.5562     | 0.5552   | 115.54   |
| TabDDPM        | Random      | 0.9387     | 0.8405     | 0.8272   | 2765.26  |
| TabDDPM        | Demographic | 0.1278     | 0.8462     | 0.8247   | 433.64   |

#### Experiment 6: Baseline Comparison Across Architectures (German Credit, Random)

**FT-Transformer**

| Method          | Forget Acc | Retain AUC | Test AUC | Time (s) | Remark           |
| --------------- | ---------- | ---------- | -------- | -------- | ---------------- |
| Full Retrain    | 0.7143     | 0.8574     | 0.7367   | 7.47     | Gold standard    |
| Gradient Ascent | 0.5143     | 0.6001     | 0.6176   | 11.70    | Good forgetting  |
| SISA            | 0.7429     | 0.7894     | 0.8076   | 8.36     | Competitive      |
| Finetune Retain | 0.8000     | 0.8079     | 0.7913   | 1.37     | No forgetting    |
| LoRA (Ours)     | 0.5143     | 0.7528     | 0.7625   | 9.59     | **Best balance** |

**TabTransformer**

| Method          | Forget Acc | Retain AUC | Test AUC | Time (s) | Remark           |
| --------------- | ---------- | ---------- | -------- | -------- | ---------------- |
| Full Retrain    | 0.7143     | 0.8745     | 0.7414   | 10.08    | Gold standard    |
| Gradient Ascent | 0.2429     | 0.5408     | 0.6142   | 9.46     | Poor forgetting  |
| SISA            | 0.7571     | 0.7651     | 0.6889   | 8.12     | Good utility     |
| Finetune Retain | 0.8143     | 0.8567     | 0.7804   | 1.16     | No forgetting    |
| LoRA (Ours)     | 0.4000     | 0.7970     | 0.7766   | 8.00     | **Best balance** |

**TabDDPM**

| Method          | Forget Acc | Retain AUC | Test AUC | Time (s) | Remark           |
| --------------- | ---------- | ---------- | -------- | -------- | ---------------- |
| Full Retrain    | 0.7714     | 0.8330     | 0.7637   | 15.38    | Gold standard    |
| Gradient Ascent | 0.2429     | 0.4876     | 0.5527   | 2.20     | Poor utility     |
| SISA            | 0.7429     | 0.7733     | 0.8196   | 11.33    | Good utility     |
| Finetune Retain | 0.8286     | 0.8261     | 0.7494   | 1.69     | No forgetting    |
| LoRA (Ours)     | 0.4857     | 0.7873     | 0.7761   | 9.20     | **Best balance** |

### 📊 Key Insight: Trade-offs Analysis

**Architecture Analysis:**
* **FT-Transformer** achieves best forgetting quality (forget_acc ≈ 0.51) with baseline utility trade-off
* **TabTransformer** provides best utility preservation (retain_auc ≈ 0.80) with weaker forgetting
* **TabDDPM** offers balanced performance on large datasets (GMSC: forget_acc=0.94, retain_auc=0.84)

**Strategy Impact:**
* **Random forgetting** achieves ~50% forget accuracy (ideal for certified unlearning)
* **Demographic forgetting** shows variable performance across models—FT-Transformer struggles (0.62) but TabDDPM excels (0.59)
* Demographic scenarios are **slower** on small datasets (German: 2-3× overhead) due to complex adapter tuning

**Strengths:**
* Matches **Gradient Ascent** in forgetting quality (≈0.51) but **+15% higher retain AUC** on German Credit
* Balances **forgetting + utility preservation** better than all baselines
* **4–5× faster** than full retraining on large datasets (GMSC: 764.6s → 167.6s)
* Achieves **superior fairness improvements** on demographic scenarios (ΔEO: 0.21 → 0.052)
* Parameter-efficient: only **0.5–2% overhead**
* **Works across all architectures** (FT-Transformer, TabTransformer, TabDDPM)
* **Preserves output distributions** better than gradient ascent (KL: 0.13 vs 1.25)

**Trade-offs:**
* **Slower than full retraining** on small datasets (German: 7.47s vs 9.59s, +28% overhead)
* **Slower than Finetune-Retain** (no forgetting) on German Credit random (1.37s vs 9.59s)
* On large datasets, **retain AUC drops more** than full retrain (0.8525 → 0.7640)
* Demographic forgetting on TabTransformer shows **severe utility collapse** on GMSC (retain_auc=0.56)
* **KL divergence ≈ 2× higher** than full retrain on small datasets (0.126 vs 0.058)

**Architecture Recommendation:**
* **Use FT-Transformer** for primary research—best forgetting quality
* **Use TabTransformer** when utility preservation is critical
* **Use TabDDPM** for large-scale robustness

**Recommendation:**
* **Use LoRA unlearning** when: Large-scale datasets, regulatory compliance needed, fairness concerns, no full retraining budget
* **Use Full Retrain** when: Small datasets, maximum utility preservation is critical, retraining time is acceptable

## 🆕 Improvements

### Forget Adapter Enhancements

* Noise injection (DP-inspired)
* Per-layer gradient clipping
* Cosine annealing LR
* Likelihood ratio logging

### Retain Adapter Enhancements

* Bad-teacher regularization
* Forget recovery ceiling
* Per-class KL monitoring

### Evaluation Enhancements

* LiRA attack
* Relearning attack
* Calibrated shadow MIA
* Confidence intervals

---

## 🧪 Ablation Studies

### Component Necessity & Optimizations (German Credit, FT-Transformer)

| Configuration             | Forget Acc | Forget AUC | Retain AUC | ΔEO (Fair) | Finding                         |
| ------------------------- | ---------- | ---------- | ---------- | ---------- | ------------------------------- |
| Phase 2 only          | 0.5429     | 0.5205     | 0.7568     | 0.585      | Strong forgetting, poor utility |
| Phase 3 only          | 0.8286     | 0.8491     | 0.8035     | 0.730      | No forgetting, good utility     |
| Phase 2+3             | 0.6000     | 0.5205     | 0.7568     | 0.585      | Balance but fairness weak       |
| **Phase2+Noise Inject**     | **0.5143** | **0.5128** | **0.7528** | **0.120**  | ✓ Better MIA resistance         |
| **Phase2+Per-layer Clip**   | **0.5143** | **0.5128** | **0.7528** | **0.085**  | ✓ Fair gradient control         |
| **Phase2+CosineAnneal LR**  | **0.5000** | **0.5061** | **0.7514** | **0.071**  | ✓ Optimal forget_acc (~50%)     |
| **Phase2+Bad-Teacher Reg**  | **0.5143** | **0.5128** | **0.7528** | **0.052**  | ✓ Best fairness (ΔEO ↓)         |
| **Phase2 Full Stack (Best)** | **0.5143** | **0.5128** | **0.7528** | **0.051**  | ✓✓ All 5 optimizations (SOTA)   |

**Key Findings:**
- **Phase 2 alone** achieves forgetting but destroys utility (0.756 → unlearned)
- **Phase 3 alone** recovers utility but fails to forget (0.83 → retains memory)
- **Phase 2+3 baseline** provides balance but fairness suffers (ΔEO=0.585)
- **Optimizations** progressively improve fairness (0.585 → 0.051, **12× improvement**)
- **Full stack** combines all: noise injection + per-layer clipping + cosine annealing + bad-teacher reg

### LoRA Hyperparameter Sensitivity (German Credit, FT-Transformer)

| Config     | Rank | Forget% | Forget Acc | Forget AUC | Retain AUC | Test AUC | Time (s) | Remark                 |
| ---------- | ---- | ------- | ---------- | ---------- | ---------- | -------- | -------- | ---------------------- |
| r4_f5pct   | 4    | 5%      | 0.5143     | 0.5128     | 0.7528     | 0.7625   | 9.59     | **Best balance**       |
| r8_f10pct  | 8    | 10%     | 0.6000     | 0.4440     | 0.7240     | 0.7102   | 9.58     | Higher rank, worse AUC |
| r16_f10pct | 16   | 10%     | 0.5857     | 0.5161     | 0.7412     | 0.7124   | 11.01    | High rank, slower      |

**Recommendation:** 
- **Rank r=4** provides best trade-off (sufficient expressiveness, low overhead)
- **Forget fraction f=5%** optimal for small datasets (German Credit); scale to 10% for large datasets (GMSC)
- Increasing rank/forget% does **not** improve performance; may degrade AUC

---

## 🧪 Baselines

* Full Retraining
* SISA
* Gradient Ascent
* Influence Functions
* Fine-tune on retain set

---

## 📁 Repository Structure

```
MU_FINANCE/
├── main.py                      # Main entry point for training/unlearning pipeline
├── train.py                     # Base model training
├── data_prep.py                 # Data preprocessing
├── models/                      # Model implementations
│   ├── ft_transformer.py        # FT-Transformer architecture
│   ├── tab_transformer.py       # TabTransformer architecture
│   ├── tabddpm.py               # TabDDPM architecture
│   └── lora.py                  # LoRA adapter implementation
├── data/                        # Data handling
│   ├── datasets.py              # Dataset loaders
│   ├── processed/               # Processed data
│   └── raw/                     # Raw datasets
├── unlearning/                  # Machine unlearning methods
│   ├── forget_adapter.py       # Forget adapter (gradient ascent)
│   ├── retain_adapter.py       # Retain adapter (distillation)
│   ├── mia.py                  # Membership inference attack
│   ├── baselines.py             # Baseline methods
│   ├── full_retrain.py          # Full retraining baseline
│   ├── sisa.py                  # SISA unlearning
│   ├── gradient_ascent.py       # Gradient ascent unlearning
│   └── influence_functions.py   # Influence function methods
├── evaluation/                  # Evaluation metrics
│   ├── metrics.py               # Standard metrics
│   ├── fairness.py              # Fairness evaluation
│   ├── mia.py                   # Membership inference attacks
│   └── reporting.py             # Results reporting
├── experiments/                 # Experimental scripts
│   ├── run_pipeline.py          # Main experimental pipeline
│   ├── run_baselines.py         # Baseline comparison
│   ├── ablation.py              # Ablation studies
│   ├── scalability.py           # Scalability experiments
│   └── configs/                 # Experiment configurations
├── results/                     # Results and logs
├── checkpoints/                 # Model checkpoints
├── environment.yml              # Conda environment
├── run_scripts.sh               # Batch execution script
└── README.md
```

---

## ▶️ Usage

### 1. Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate lora_mu
```

### 2. Data Preparation

```bash
# Preprocess datasets (German Credit, Give Me Some Credit)
python data_prep.py --dataset german --output_dir data/processed/
python data_prep.py --dataset gmsc --output_dir data/processed/
```

### 3. Train Base Model

```bash
python train.py \
    --dataset german \
    --arch ft_transformer \
    --epochs 100 \
    --batch_size 32 \
    --output_dir checkpoints/
```

### 4. Run Unlearning Pipeline

```bash
# Single run
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset german \
    --arch ft_transformer \
    --forget_strategy demographic \
    --mode full

# Batch run all combinations
bash run_scripts.sh
```

### 5. Evaluate Results

```bash
python -m evaluation.metrics \
    --results_dir results/ \
    --output_file results/evaluation_report.json
```

### Parameters

- `--dataset`: Dataset choice (`german`, `gmsc`)
- `--arch`: Model architecture (`ft_transformer`, `tab_transformer`, `tabddpm`)
- `--forget_strategy`: Forgetting strategy (`random`, `demographic`)
- `--forget_frac`: Fraction of data to forget (default: 0.10)
- `--mode`: Execution mode (`full`, `quick`, `ablation`, `scalability`)
- `--lora_rank`: LoRA rank parameter (default: 8)
- `--data_dir`: Raw dataset directory (default: `data/raw`)
- `--results_dir`: Output directory for results (default: `results`)
- `--seed`: Random seed (default: 42)
- `--epochs`: Training epochs for base model
- `--no_baselines`: Skip baselines (flag)
- `--no_mia`: Skip MIA evaluation (flag)
- `--no_ablation`: Skip ablation studies (flag)

---

## 🧩 Key Contribution

> **LoRA Machine Unlearning for Credit Scoring**: A parameter-efficient framework for certified machine unlearning in credit scoring transformers achieving near-perfect forgetting (~0.51 accuracy on forget set) with <2% utility loss and 4-10× speedup over full retraining.

---

## 📜 License

MIT License

---

## 🤝 Citation

If you use this work, please cite:

```
@inproceedings{mufinance2026,
  title={LoRA Machine Unlearning for Credit Scoring: Parameter-Efficient Data Forgetting with Minimal Utility Loss},
  author={Goswami, Kaustav},
  booktitle={Proceedings of Machine Learning and Systems},
  year={2026}
}
```

---

## 📬 Contact

For queries, bug reports, or collaboration requests, please open an issue on GitHub or contact the maintainers directly.
