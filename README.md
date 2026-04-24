# AI-Driven Personalized Learning Content Recommendation for Dyslexic Students

> A full machine learning pipeline that classifies students into personalized learning tiers based on cognitive and behavioral scores — going beyond binary dyslexia detection to actionable content recommendations.

**Authors:** Akanksha Semwal, Vikirna Majumdar, Darshana Krishna  
**Institution:** SRM Institute of Science and Technology, Ghaziabad  
**Supervisor:** Dr. Naresh Sharma, Assistant Professor

---

## What This Project Does

Most dyslexia AI research only answers: *"Does this student have dyslexia?"*

This pipeline answers: *"What kind of learning content does this student need?"*

It classifies students into one of four content recommendation tiers:

| Tier | Score Range | Description |
|------|-------------|-------------|
| Intensive Support | 0 – 12 | Students requiring strong structured intervention |
| Basic Learning | 12 – 18 | Students needing guided support |
| Standard Learning | 18 – 24 | Students needing moderate guidance |
| Advanced Learning | 24 – 31 | Students ready for challenging content |

Classification is based on three key cognitive scores: **Reading Ability**, **Memory Score**, and **Attention Score**.

---

## Key Results

- **Best Model:** Neural Network (MLP Classifier)
- **Accuracy:** 96.79%
- **R² Score:** 0.8935
- All 8 classifiers evaluated across 6 metrics: Accuracy, F1-Score, ROC-AUC, RMSE, MAE, R²

---

## Pipeline Overview

```
Dataset Loading
      ↓
Data Exploration (500 student records, 13 features)
      ↓
Data Cleaning (IQR outlier capping, label encoding, StandardScaler)
      ↓
Exploratory Data Analysis (histograms, correlation heatmap, box plots)
      ↓
Feature Selection — 4 parallel methods:
    ├── Filter: ANOVA F-Score (SelectKBest)
    ├── Wrapper: Recursive Feature Elimination (RFE)
    ├── Embedded: Random Forest Importance
    └── Embedded: Lasso (L1 Regularization)
      ↓
SMOTE Oversampling (class balance)
      ↓
Model Training — 8 classifiers:
    LR | RF | SVM | NN | XGBoost | LightGBM | CatBoost | GBM
      ↓
Evaluation (Accuracy, F1, ROC-AUC, RMSE, MAE, R²)
      ↓
Feature Importance + Content Recommendation Output
```

---

## Feature Selection Summary

4 methods were applied in parallel. A feature was kept only if selected by ≥ 2 methods (consensus voting).

| Feature | Filter | Wrapper | RF | Lasso | Selected |
|---------|--------|---------|-----|-------|----------|
| Reading_Ability | ✅ | ✅ | ✅ | ✅ | **YES** |
| Attention_Score | ✅ | ✅ | ✅ | ✅ | **YES** |
| Memory_Score | ✅ | ✅ | ✅ | ✅ | **YES** |
| Dyslexia_Risk | ✅ | ✅ | ❌ | ❌ | **YES** |
| Audio_Score | ✅ | ❌ | ❌ | ❌ | No |
| Age, NativeLang, etc. | ❌ | Partial | ❌ | ❌ | No |

---

## Dataset

- **Size:** 500 synthetic student records
- **Source:** Modeled on the [Kaggle Dyslexia Dataset](https://www.kaggle.com/datasets/luzrello/dyslexia)
- **Features:** 13 columns including Age, Gender, Memory Score, Reading Ability, Attention Score, Spelling Ability, Visual Score, Speed Score, Audio Score, and more
- **Target:** `Content_Recommendation` (4-class) and `Dyslexia_Risk` (binary)
- **Class imbalance handled with:** SMOTE (Synthetic Minority Over-sampling Technique)

---

## Tech Stack

```
Python 3.x
├── pandas, numpy          — data handling
├── scikit-learn           — ML models, feature selection, preprocessing
├── imbalanced-learn       — SMOTE
├── xgboost, lightgbm      — gradient boosting
├── catboost               — gradient boosting
├── matplotlib, seaborn    — visualization
└── scipy                  — statistical methods
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/dyslexia-ai-support.git
cd dyslexia-ai-support
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook dyslexia_updated_pipeline.ipynb
```

> If the real dataset (`dyslexia.csv`) is not found, the notebook automatically generates a synthetic dataset of 500 students and saves it as `dyslexia_synthetic.csv`.

---

## Requirements

Create a `requirements.txt` with:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
catboost
scipy
jupyter
```

Install with:
```bash
pip install -r requirements.txt
```

---

## What Makes This Different

| Gap in Prior Work | How This Pipeline Addresses It |
|---|---|
| Most studies do binary detection only | 4-class content recommendation output |
| Feature selection rarely applied systematically | 4 parallel methods with consensus voting |
| Few papers compare more than 2–3 classifiers | 8 classifiers compared with 6 evaluation metrics |
| Class imbalance often ignored | SMOTE applied before train/test split |
| Accuracy used alone | ROC-AUC, F1, RMSE, MAE, R² all included |
| Black-box models with no explainability | RF feature importance + permutation importance |

---

## Research Paper

This project is accompanied by a research paper:

**"AI-Driven Personalized Learning Content Recommendation for Dyslexic Students: A Machine Learning Pipeline Analysis"**  
Semwal A., Majumdar V., Krishna D. (2025)  
SRM Institute of Science and Technology

---

## License

This project is for academic and research purposes.
