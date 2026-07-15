# Mini Project #02

**EECE 6544: Introduction to Machine Learning and Pattern Recognition (Summer 2026)**

A junior data-science engagement for **HealthGuard Insurance**. The company prices plans from outdated
actuarial tables, causing it to **underprice high-risk customers** (losses) and **overprice low-risk
customers** (churn). This project uses the company's historical customer records to:

1. **Understand what drives medical costs** — exploratory data analysis (EDA).
2. **Predict annual medical charges** — regression (7 models compared).
3. **Flag "expensive" customers** — classification (charges above the median).

Full write-up: **[REPORT.md](REPORT.md)**.

---

## What the project does

- Loads and profiles the raw insurance dataset, then cleans it with the full pandas wrangling toolkit
  (creating, selecting, filtering, sorting, replacing, renaming, missing-value handling, dropping
  rows/columns/duplicates, grouping, aggregating, applying functions, resampling, concatenating,
  merging + type conversion, binning, encoding).
- Produces **10 board-ready visualizations** and written insights.
- Builds, tunes, and compares **7 regression models** (MAE / MSE / RMSE / R²) and recommends one.
- Builds and compares **5 classifiers** (accuracy / precision / recall / F1 / ROC-AUC) and recommends one.

---

## Repository structure

```
.
├── README.md                    # this file
├── REPORT.md                    # technical report (cleaning, EDA, models, recommendations)
├── requirements.txt             # pinned Python dependencies
├── notebooks/
│   └── insurance_analysis.ipynb # end-to-end notebook (cleaning + EDA + regression + classification)
├── data/
│   ├── raw/insurance.csv        # raw Kaggle dataset (see below)
│   └── processed/               # cleaned data + metric tables (generated)
└── figures/                     # all generated charts (generated)
```

---

## How to download the dataset from Kaggle

A copy of `insurance.csv` is included at `data/raw/insurance.csv` so the notebook runs out of the box.
To obtain it fresh from the original source, use either option below and place it at
`data/raw/insurance.csv`.

**Option A — Kaggle CLI**
```bash
pip install kaggle
# Put your kaggle.json API token in ~/.kaggle/ (Kaggle → Account → Create New API Token)
kaggle datasets download -d mirichoi0218/insurance -p data/raw --unzip
```

**Option B — Manual**
1. Visit https://www.kaggle.com/datasets/mirichoi0218/insurance
2. Download `insurance.csv`.
3. Save it to `data/raw/insurance.csv`.

The file has 1,338 rows and columns: `age, sex, bmi, children, smoker, region, charges`.

---

## How to run

```bash
# 1. (optional) create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. ensure the dataset is at data/raw/insurance.csv (see above)

# 4a. run interactively
jupyter notebook notebooks/insurance_analysis.ipynb

# 4b. or execute headless (regenerates all figures + processed CSVs)
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/insurance_analysis.ipynb
```

Running the notebook regenerates `data/processed/insurance_cleaned.csv`, the metric tables
(`regression_results.csv`, `classification_results.csv`), and all charts in `figures/`.

---

## Summary of findings

- **Smoking is the dominant cost driver:** smokers average **~$32,050/yr vs ~$8,441/yr** for
  non-smokers (~4×), and it has the highest correlation with charges (~0.79).
- **Age** raises costs steadily; **BMI** matters mostly *for smokers*, where obesity (BMI ≥ 30)
  sharply compounds cost — the **smoking × obesity** interaction is the most expensive combination.
- **Sex, region, and number of children** have small effects.
- **Best regression model — tuned Decision Tree Regressor:** R² = **0.90**, RMSE ≈ **$4,346**,
  MAE ≈ **$2,621** on held-out data, and interpretable enough to defend to the board.
- **Best classifier — SVM (RBF):** F1 = **0.939**, ROC-AUC = **0.964**, accuracy = **94%** for flagging
  above-median-cost customers (median charge = **$9,386**).
- **Pricing implication:** premiums should explicitly load for **smoking status** and, for smokers,
  **BMI** — directly addressing the under/over-pricing problems.

See **[REPORT.md](REPORT.md)** for the full methodology, model-by-model notes, and figures.
