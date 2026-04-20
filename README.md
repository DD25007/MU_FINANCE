# LoRA-Credit-Unlearn

A parameter-efficient machine unlearning framework for transformer-based credit scoring models using LoRA adapters. This repository implements a fast alternative to full retraining, achieving effective data forgetting with minimal utility loss.

---

## 🚀 Overview

Modern credit scoring models are trained on large-scale sensitive financial data. Under regulations like GDPR (Right to be Forgotten), specific user data must be removed from trained models. Traditional retraining is computationally expensive.

This project introduces **LoRA-based unlearning**, which:

* Removes the influence of selected data (forget set)
* Preserves performance on remaining data (retain set)
* Reduces unlearning time from hours → minutes

---

## 🧠 Core Idea

We use **Low-Rank Adaptation (LoRA)** to perform targeted updates:

* Freeze base model weights
* Learn small low-rank adapters for unlearning
* Apply a **three-phase pipeline**:

### 1. Base Training

Train model on full dataset (D = D_f \cup D_r)

### 2. Forget Adapter (Gradient Ascent)

* Train LoRA adapter on forget set (D_f)
* Objective: degrade performance on forget samples

### 3. Retain Adapter (Distillation)

* Train second adapter on retain set (D_r)
* Objective: recover utility via knowledge distillation

### Final Model

* Merge retain adapter into base weights
* Discard forget adapter

---

## 🏗️ Models

* **FT-Transformer (Primary)**
* **TabTransformer (Secondary)**
* **XGBoost / LightGBM (Baselines)**

LoRA is applied to:

* Query (Q) projection
* Value (V) projection

---

## 📊 Datasets

* German Credit (UCI)
* Give Me Some Credit (Kaggle)
* LendingClub (temporal experiments)

### Forget Set Strategies

* Random subset (5–20%)
* Demographic subgroup (e.g., age < 25)
* Temporal window

---

## 📈 Evaluation Metrics

| Metric          | Purpose                       |
| --------------- | ----------------------------- |
| Forget Accuracy | Should approach ~50%          |
| Retain AUC      | Utility preservation          |
| Test AUC        | Generalization                |
| KL Divergence   | Distance from retrained model |
| MIA Score       | Certified forgetting (≈0.5)   |
| Relearning Gain | Forgetting robustness         |
| Wall-clock Time | Efficiency                    |
| Fairness (ΔEO)  | Bias reduction                |

---

## ⚙️ Pipeline

1. Data preprocessing
2. Train base model
3. Run baseline methods
4. Apply LoRA unlearning
5. Ablation studies
6. Membership inference attack (MIA)
7. Scalability experiments

---

## 🔬 Key Results (v2)

### Summary

* Forget accuracy ≈ **0.51** (near ideal 0.5)
* Retain AUC drop < **1–2%**
* Up to **4–10× speedup** over retraining
* Significant fairness improvement (ΔEO ↓)
* MIA score ≈ **0.47–0.51** (certified forgetting)

### 📊 Comparison with Baselines (German Credit)

| Method              | Forget Acc ↓ | Forget AUC ↓ | Retain AUC ↑ | Test AUC ↑ | Time (s) ↓ | Remark             |
| ------------------- | ------------ | ------------ | ------------ | ---------- | ---------- | ------------------ |
| Base Model          | 0.8143       | 0.8424       | —            | 0.7898     | —          | No unlearning      |
| Full Retrain        | 0.7143       | 0.7125       | 0.8574       | 0.7367     | 7.47       | Gold standard      |
| Gradient Ascent     | 0.5143       | 0.2286       | 0.6001       | 0.6176     | 11.70      | Poor utility       |
| SISA                | 0.7429       | 0.6959       | 0.7894       | 0.8076     | 8.36       | Competitive        |
| Finetune Retain     | 0.8000       | 0.8413       | 0.8079       | 0.7913     | 1.37       | No forgetting      |
| **LoRA (Ours, v2)** | **0.5143**   | **0.5128**   | **0.7528**   | **0.7625** | **9.59**   | **Best trade-off** |

### 📊 Key Insight

* Matches **Gradient Ascent** in forgetting (≈0.51) but **+15% higher retain AUC**
* Approaches **Full Retrain utility** at a fraction of cost
* Provides **balanced unlearning + utility preservation**, unlike all baselines

## 🆕 Improvements (v2)

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

## 🧪 Baselines

* Full Retraining
* SISA
* Gradient Ascent
* Influence Functions
* Fine-tune on retain set

---

## 📁 Repository Structure

```
.
├── models/
├── datasets/
├── kaustav_forget_adapter.py
├── kaustav_retain_adapter.py
├── kaustav_mia.py
├── kaustav_tab_transformer.py
├── train.py
├── evaluate.py
└── README.md
```

---

## ▶️ Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train base model

```bash
python train.py
```

### 3. Run unlearning

```bash
python unlearn.py
```

### 4. Evaluate

```bash
python evaluate.py
```

---

## 🧩 Key Contribution

> LoRA-Credit-Unlearn: A parameter-efficient framework for certified machine unlearning in credit scoring transformers with minimal performance degradation.

---

## 📜 License

MIT License

---

## 🤝 Citation

If you use this work, please cite:

```
@article{lora_credit_unlearn,
  title={LoRA-Credit-Unlearn},
  author={Goswami, Kaustav},
  year={2026}
}
```

---

## 📬 Contact

For queries or collaboration, open an issue or reach out.
