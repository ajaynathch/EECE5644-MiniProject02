# Technical Report

**Course:** EECE 5644: Introduction to Machine Learning and Pattern Recognition (Summer 2026)
**Project:** Mini Project #02
**Dataset:** [Kaggle — mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance) (1,338 customer records)

All results below are produced reproducibly by [notebooks/insurance_analysis.ipynb](notebooks/insurance_analysis.ipynb).

---

## 1. Business problem → ML tasks

HealthGuard prices plans from outdated actuarial tables, causing it to **underprice high-risk
customers** (losses) and **overprice low-risk customers** (churn). Ms. Nabil's three requirements were
translated into ML tasks:

| Requirement (her words) | ML task |
|---|---|
| "Understand our customers first" | Exploratory Data Analysis |
| "Predict the medical charges" | Regression (7 models) |
| "Flag the expensive customers" | Binary classification (charges > median) |

---

## 2. Data cleaning: decisions & rationale

The raw file is treated as **immutable**; every transformation returns a new dataframe, and the
cleaned result is written to `data/processed/insurance_cleaned.csv` (1,337 rows × 11 columns after
cleaning + feature engineering).

| Step | Decision | Rationale |
|---|---|---|
| Profiling | `head/shape/info/describe/value_counts` | Confirmed 1,338×7, dtypes, and category spellings. |
| Duplicates | Dropped the **1** exact duplicate row | A repeated customer would double-weight one record and bias every estimate. |
| Missing values | Median (numeric) / mode (categorical) | Median resists the heavy right-skew in `charges`. The raw file had none, but the guard keeps the pipeline safe for future, messier extracts. |
| Category hygiene | strip + lowercase + remap (`y→yes`, etc.) | Prevents `"Yes"`/`"yes"` splitting into separate categories. |
| Validity filter | age∈[18,100], bmi∈[10,60], charges>0 | Removes impossible/data-entry values. None triggered here — it documents our quality bar. |
| Type enforcement | `astype` int/float/category | Correct dtypes for efficient, correct modeling. |
| Feature engineering | `bmi_category`, `age_group`, `smoker_flag`, regional `cost_index` | Converts raw fields into business-meaningful signals. |

**pandas techniques demonstrated (per the brief):** creating, selecting, filtering, sorting,
replacing, renaming, handling missing values, dropping rows/columns/duplicates, grouping,
aggregating, applying functions, resampling (on a synthesised enrollment date), concatenating, and
merging (an external region lookup) — plus type conversion, binning, and one-hot encoding.

---

## 3. Requirement 1: Exploratory Data Analysis

Ten figures are saved to `figures/`. Key ones:

| Figure | Insight |
|---|---|
| `fig1_charges_distribution.png` | `charges` is strongly right-skewed; log-transform is near-normal. |
| `fig2_charges_by_smoker.png` | **Smokers ≈ $32,050/yr vs non-smokers ≈ $8,441/yr — about 4×.** |
| `fig3_charges_vs_age.png` | Charges rise steadily with age, in three separated bands. |
| `fig4_charges_vs_bmi.png` | BMI barely affects non-smokers, but sharply raises smokers' costs past BMI 30. |
| `fig5_correlation_heatmap.png` | Smoking has the highest correlation with charges (~0.79); age next. |
| `fig7_smoker_bmi_interaction.png` | The **smoking × obesity** interaction is the single most expensive combination. |

**Headline insights for the board**

1. **Smoking is the dominant cost driver (~4×).**
2. **Age raises costs steadily.**
3. **BMI matters chiefly *for smokers*** — obesity (BMI ≥ 30) compounds smoking cost.
4. **Sex, region, and number of children have small effects.**
5. **Surprising pattern:** costs form *stratified layers* (by smoking, then obesity) rather than a
   smooth cloud — a pricing model must treat smoking as a first-class factor.

---

## 4. Requirement 2: Regression (predicting charges)

Single 80/20 train/test split reused across all models. Numeric features scaled for
linear/SVR/polynomial models; trees use raw features. Categoricals one-hot encoded.

### Model comparison (held-out test set)

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| **7. Decision Tree (tuned)** | **2,621** | 1.89e7 | **4,346** | **0.897** |
| 3. Polynomial (degree 2) | 2,867 | 2.16e7 | 4,646 | 0.883 |
| 6. SVR (rbf kernel) | 1,973 | 2.41e7 | 4,909 | 0.869 |
| 2. Multiple Linear (all features) | 4,177 | 3.55e7 | 5,956 | 0.807 |
| 4. Ridge (tuned) | 4,194 | 3.57e7 | 5,972 | 0.806 |
| 5. Lasso (tuned) | 4,224 | 3.63e7 | 6,028 | 0.802 |
| 6. SVR (linear kernel) | 3,111 | 3.98e7 | 6,312 | 0.783 |
| 1. Simple Linear (smoker only) | 5,831 | 6.00e7 | 7,749 | 0.673 |

*(Regenerated to `data/processed/regression_results.csv`; comparison chart: `figures/fig9_model_comparison.png`.)*

**Notes per model**

- **Simple Linear (smoker only):** even one feature explains R²≈0.67 — confirms smoking's dominance.
- **Polynomial (2–4):** train R² keeps rising while test R² plateaus and the train–test gap widens →
  classic **overfitting**; degree 2 captures useful curvature, degrees 3–4 fit noise.
- **Ridge/Lasso:** L2/L1 regularisation, alpha tuned by 5-fold CV. Lasso's eliminated features (coef=0)
  are reported in the notebook; the strong signals (smoker, age, bmi) always survive.
- **SVR:** two kernels; RBF (scaled) strongly outperforms linear.
- **Decision Tree:** `max_depth`/`min_samples_leaf` tuned by CV; a depth-3 tree is visualised
  (`figures/fig8_decision_tree.png`) — it splits on **smoking first**, then age/BMI.

### Recommendation (regression)

**Recommend the tuned Decision Tree Regressor.** It has the best held-out accuracy
(R² = 0.90, RMSE ≈ $4,346, MAE ≈ $2,621) *and* is directly interpretable — the tree shows the pricing
logic (smoking → age/BMI) that Ms. Nabil can defend to the board and regulators. SVR-RBF is a close
accuracy alternative but is a black box; multiple linear regression is the transparent fallback.
Avoid high-degree polynomials (overfit) and the linear-kernel SVR (weakest fit).

---

## 5. Requirement 3: Classification (flag expensive customers)

Target: `expensive = 1 if charges > median`. **Median = $9,386.16**, giving a balanced ~50/50 problem.
`medical_charges` is excluded from the features to avoid label leakage. Stratified 80/20 split.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **SVM (RBF)** | **0.940** | 0.961 | 0.918 | **0.939** | **0.964** |
| Decision Tree | 0.940 | 0.992 | 0.888 | 0.937 | 0.945 |
| Random Forest | 0.937 | 0.961 | 0.910 | 0.935 | 0.951 |
| Logistic Regression | 0.907 | 0.898 | 0.918 | 0.908 | 0.953 |
| K-Nearest Neighbors | 0.858 | 0.914 | 0.791 | 0.848 | 0.924 |

*(Saved to `data/processed/classification_results.csv`; confusion matrix + ROC curves in
`figures/fig10_classification_eval.png`.)*

### Recommendation (classification)

**Recommend SVM (RBF)** — best F1 (0.939) and ROC-AUC (0.964), balancing false alarms (over-pricing)
against misses (under-pricing). Because classes are balanced, accuracy is also meaningful. The Decision
Tree is a strong, interpretable runner-up (highest precision, 0.99). Operationally, route
"expensive"-flagged applicants to manual actuarial review before quoting.

---

## 6. Summary for Ms. Nabil (non-technical)

1. **What drives cost:** smoking (~4×), then age, then obesity — and smoking × obesity is the priciest
   combination. Sex, region, and children barely matter.
2. **Predicting the dollar amount:** the Decision Tree estimates a customer's annual charges within
   about **±$2,600 on average**, and its logic is explainable.
3. **Flagging expensive customers:** the SVM classifier separates above-median customers with **94%
   accuracy / 0.96 AUC** — a fast triage signal when the exact figure isn't needed.
4. **Pricing implication:** premiums should explicitly load for **smoking status** and, for smokers,
   **BMI**. This directly fixes the under-pricing of high-risk customers and the over-pricing of
   low-risk ones that Ms. Nabil identified.

---

## 7. Reproducibility

```bash
pip install -r requirements.txt
# place the Kaggle insurance.csv at data/raw/insurance.csv (see README)
jupyter nbconvert --to notebook --execute --inplace notebooks/insurance_analysis.ipynb
```

Outputs: cleaned data → `data/processed/`, figures → `figures/`, metric tables →
`data/processed/*_results.csv`.
