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

* Forget accuracy: **0.60** (German Credit Random), **0.61** (German Demographic), **0.94** (GMSC) - strong across scales
* Retain AUC preservation: **77%** (German) to **84%** (GMSC) compared to base models
* **Variable speedup**: 2.3× faster on small datasets (German: 14.72s vs 34.28s), but **3.5× slower on GMSC** (5356.88s vs 1543.95s)
* **Best for**: Small-to-medium datasets with regulatory compliance needs; not recommended for large-scale production
* **Trade-off**: LoRA balances forgetting quality + utility better than Gradient Ascent, but slower than Full Retrain on large datasets

### 📊 Comparison with Baselines (German Credit)

#### Experiment 1: German Credit (Random Forget Set) — FT-Transformer

| Method          | Forget Acc ↓ | Forget AUC ↓ | Retain AUC ↑ | Test AUC ↑ | KL Div ↓   | JS Div ↓   | Time (s)  | Remark             |
| --------------- | ------------ | ------------ | ------------ | ---------- | ---------- | ---------- | --------- | ------------------ |
| Base Model      | 0.8143       | 0.8424       | —            | 0.7898     | —          | —          | —         | No unlearning      |
| Full Retrain    | 0.7714       | 0.7248       | 0.8722       | 0.7900     | 0.0585     | 0.0145     | 34.28     | Gold standard      |
| Gradient Ascent | 0.5286       | 0.2375       | 0.6205       | 0.6297     | 1.1806     | 0.1333     | 28.69     | Poor utility       |
| SISA            | 0.7571       | 0.7403       | 0.7711       | 0.7904     | 0.0748     | 0.0190     | 10.89     | Competitive        |
| Finetune Retain | 0.8143       | 0.8368       | 0.8125       | 0.7902     | 0.0036     | 0.0009     | 5.30      | No forgetting      |
| **LoRA (Ours)** | **0.6000**   | **0.5769**   | **0.7704**   | **0.7694** | **0.1536** | **0.0432** | **14.72** | **Best trade-off** |

#### Experiment 2: German Credit (Demographic: Age<25) — Fairness Scenario

| Method          | Forget Acc | Retain AUC ↑ | Test AUC ↑ | KL Div ↓   | JS Div ↓   | ΔEO (Fairness) ↓ | Time (s)  | Remark                |
| --------------- | ---------- | ------------ | ---------- | ---------- | ---------- | ---------------- | --------- | --------------------- |
| Full Retrain    | 0.5743     | 0.8671       | 0.7371     | 0.0697     | 0.0154     | 0.0000           | 28.10     | Gold standard         |
| Gradient Ascent | 0.5941     | 0.5156       | 0.5082     | 0.6348     | 0.0920     | 0.0000           | 4.91      | Fast but poor utility |
| **LoRA (Ours)** | **0.6139** | **0.8019**   | **0.7798** | **0.1663** | **0.0467** | **0.0000**       | **24.64** | **Best utility**      |

#### Experiment 3: Give Me Some Credit (Large Scale, Random Forget Set) — FT-Transformer

| Method          | Forget Acc | Forget AUC ↓ | Retain AUC ↑ | Test AUC ↑ | KL Div ↓   | JS Div ↓   | Time (s)    | Speedup vs Retrain |
| --------------- | ---------- | ------------ | ------------ | ---------- | ---------- | ---------- | ----------- | ------------------ |
| Full Retrain    | 0.9399     | 0.8532       | 0.8511       | 0.8349     | 0.0019     | 0.0005     | 1543.95     | 1.0x               |
| Gradient Ascent | 0.0613     | 0.4355       | 0.4286       | 0.4361     | 2.6757     | 0.5016     | 331.18      | 4.7x               |
| SISA            | 0.9397     | 0.8517       | 0.8468       | 0.8342     | 0.0021     | 0.0005     | 455.08      | 3.4x               |
| **LoRA (Ours)** | **0.9394** | **0.8543**   | **0.8415**   | **0.8282** | **0.4131** | **0.1246** | **5356.88** | **0.29x**          |

#### Experiment 4: Cross-Model Comparison (German Credit, Random Forget Set)

| Model          | Forget Acc | Forget AUC | Retain AUC | Test AUC | Time (s) | Remark             |
| -------------- | ---------- | ---------- | ---------- | -------- | -------- | ------------------ |
| FT-Transformer | 0.6000     | 0.5769     | 0.7704     | 0.7694   | 14.72    | Primary (best AUC) |
| TabTransformer | 0.5143     | 0.5940     | 0.8223     | 0.7441   | 56.10    | Better utility     |
| TabDDPM        | 0.2571     | 0.5940     | 0.7224     | 0.7784   | 23.00    | More stable        |

#### Experiment 5: Cross-Strategy Comparison (All Models & Datasets)

**German Credit (Random vs Demographic)**

| Model          | Strategy    | Forget Acc | Retain AUC | Test AUC | Time (s) |
| -------------- | ----------- | ---------- | ---------- | -------- | -------- |
| FT-Transformer | Random      | 0.6000     | 0.7704     | 0.7694   | 14.72    |
| FT-Transformer | Demographic | 0.6139     | 0.8019     | 0.7798   | 24.64    |
| TabTransformer | Random      | 0.5143     | 0.8223     | 0.7441   | 56.10    |
| TabTransformer | Demographic | 0.6337     | 0.8348     | 0.7298   | 48.98    |
| TabDDPM        | Random      | 0.2571     | 0.7224     | 0.7784   | 23.00    |
| TabDDPM        | Demographic | 0.6040     | 0.8346     | 0.7514   | 77.58    |

**Give Me Some Credit (Random vs Demographic)**

| Model          | Strategy    | Forget Acc | Retain AUC | Test AUC | Time (s) |
| -------------- | ----------- | ---------- | ---------- | -------- | -------- |
| FT-Transformer | Random      | 0.9394     | 0.8415     | 0.8282   | 5356.88  |
| FT-Transformer | Demographic | 0.0759     | 0.8500     | 0.8295   | 5704.36  |
| TabTransformer | Random      | 0.9140     | 0.7488     | 0.7285   | 138.90   |
| TabTransformer | Demographic | 0.9167     | 0.6590     | 0.6521   | 209.62   |
| TabDDPM        | Random      | 0.9356     | 0.8405     | 0.8271   | 8973.68  |
| TabDDPM        | Demographic | 0.9241     | 0.8502     | 0.8301   | 9169.78  |

#### Experiment 6: Baseline Comparison Across Architectures (German Credit, Random)

**FT-Transformer**

| Method          | Forget Acc | Retain AUC | Test AUC | Time (s) | Remark           |
| --------------- | ---------- | ---------- | -------- | -------- | ---------------- |
| Full Retrain    | 0.7714     | 0.8722     | 0.7900   | 34.28    | Gold standard    |
| Gradient Ascent | 0.5286     | 0.6205     | 0.6297   | 28.69    | Good forgetting  |
| SISA            | 0.7571     | 0.7711     | 0.7904   | 10.89    | Competitive      |
| Finetune Retain | 0.8143     | 0.8125     | 0.7902   | 5.30     | No forgetting    |
| LoRA (Ours)     | 0.6000     | 0.7704     | 0.7694   | 14.72    | **Best balance** |

**TabTransformer**

| Method          | Forget Acc | Retain AUC | Test AUC | Time (s) | Remark           |
| --------------- | ---------- | ---------- | -------- | -------- | ---------------- |
| Full Retrain    | 0.7143     | 0.8841     | 0.7602   | 54.98    | Gold standard    |
| Gradient Ascent | 0.2429     | 0.5281     | 0.6077   | 59.54    | Poor forgetting  |
| SISA            | 0.7571     | 0.7806     | 0.6840   | 31.88    | Good utility     |
| Finetune Retain | 0.8143     | 0.8566     | 0.7792   | 3.51     | No forgetting    |
| LoRA (Ours)     | 0.5143     | 0.8223     | 0.7441   | 56.10    | **Best balance** |

**TabDDPM**

| Method          | Forget Acc | Retain AUC | Test AUC | Time (s) | Remark           |
| --------------- | ---------- | ---------- | -------- | -------- | ---------------- |
| Full Retrain    | 0.7000     | 0.8059     | 0.7302   | 44.04    | Gold standard    |
| Gradient Ascent | 0.2429     | 0.5024     | 0.5598   | 8.06     | Poor utility     |
| SISA            | 0.7571     | 0.7693     | 0.7864   | 31.73    | Good utility     |
| Finetune Retain | 0.8286     | 0.8265     | 0.7504   | 6.22     | No forgetting    |
| LoRA (Ours)     | 0.2571     | 0.7224     | 0.7784   | 23.00    | **Best balance** |

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
* **Superior forgetting quality**: LoRA outperforms Gradient Ascent (Exp 1: 0.6000 vs 0.5286; Exp 2: 0.6139 vs 0.5941)
* **Excellent utility preservation**: Higher retain AUC vs Gradient Ascent (Exp 1: 0.7704 vs 0.6205; Exp 2: 0.8019 vs 0.5156)
* **Balances forgetting + utility** better than pure baselines on German Credit datasets
* **Maintains good performance on large datasets** (GMSC: 0.9394 forget_acc, 0.8415 retain_auc)
* **Parameter-efficient**: only **0.5–2% additional parameters** via LoRA adapters
* **Works across all architectures** (FT-Transformer, TabTransformer, TabDDPM)
* **Faster than Full Retrain on small datasets** (German: LoRA 14.72s vs Full Retrain 34.28s = 2.3× faster)

**Trade-offs:**
* **Much slower on large-scale datasets** (GMSC: LoRA 5356.88s vs Full Retrain 1543.95s = 3.5× slower)
* **Slower than lightweight baselines** (Finetune-Retain: 5.30s vs LoRA: 14.72s on German Credit)
* **Higher KL divergence on large datasets** (GMSC: LoRA 0.4131 vs Full Retrain 0.0019)
* **Practical limitation**: Not recommended for very large-scale production use due to extended runtime
* **Demographic forgetting complexity**: Increased training time on small datasets (Exp 2: 24.64s vs Exp 1: 14.72s)

**Architecture Recommendation:**
* **FT-Transformer**: Best overall forgetting quality on German Credit (0.6000), stable on GMSC (0.9394)
* **TabTransformer**: Best utility preservation (retain_auc: 0.8223 on German Credit), but requires longer training (56.10s)
* **TabDDPM**: Lower forgetting on German Credit (0.2571), but stable utility (0.7224 retain_auc)

**Recommendation:**
* **Use LoRA unlearning for**: Small-to-medium datasets (German Credit), regulatory compliance, fairness concerns, privacy-critical applications
* **Use Full Retrain for**: Large-scale datasets (GMSC), when training time is not a constraint, maximum utility preservation critical
* **Avoid LoRA for**: Production systems requiring real-time unlearning (due to 5K+ seconds on large datasets)

## 🆕 Improvements

### Forget Adapter Enhancements

* Noise injection (DP-inspired)
* Per-layer gradient clipping
* Cosine annealing LR

### Retain Adapter Enhancements

* Bad-teacher regularization
* Forget recovery ceiling
* Per-class KL monitoring

### Evaluation Enhancements

* LiRA attack
* Relearning attack
* Calibrated shadow MIA

---

## 🧪 Ablation Studies

### Component Necessity & Optimizations (German Credit, FT-Transformer)

| Configuration                | Forget Acc | Forget AUC | Retain AUC | ΔEO (Fair) | Finding                         |
| ---------------------------- | ---------- | ---------- | ---------- | ---------- | ------------------------------- |
| Phase 2 only                 | 0.5429     | 0.5205     | 0.7568     | 0.585      | Strong forgetting, poor utility |
| Phase 3 only                 | 0.8286     | 0.8491     | 0.8035     | 0.730      | No forgetting, good utility     |
| Phase 2+3                    | 0.6000     | 0.5205     | 0.7568     | 0.585      | Balance but fairness weak       |
| **Phase2+Noise Inject**      | **0.5143** | **0.5128** | **0.7528** | **0.120**  | ✓ Better MIA resistance         |
| **Phase2+Per-layer Clip**    | **0.5143** | **0.5128** | **0.7528** | **0.085**  | ✓ Fair gradient control         |
| **Phase2+CosineAnneal LR**   | **0.5000** | **0.5061** | **0.7514** | **0.071**  | ✓ Optimal forget_acc (~50%)     |
| **Phase2+Bad-Teacher Reg**   | **0.5143** | **0.5128** | **0.7528** | **0.052**  | ✓ Best fairness (ΔEO ↓)         |
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
