# Customer Churn Prediction

A machine learning pipeline that predicts which telecom customers are likely to cancel their subscription next month, so a retention team can proactively reach out before they leave.

## Problem Statement

A subscription telecom company wants to flag customers likely to churn so the retention team can intervene early. The model must not just classify "will churn / won't churn" — it needs to rank customers by risk and account for the real cost tradeoff: a missed churner (false negative) is far more costly than a wasted retention offer (false positive).

## Results Summary

Three models were trained and compared: Logistic Regression, Decision Tree, and Random Forest.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** ✅ | 0.558 | **0.801** | 0.657 | **0.856** |
| Random Forest | 0.715 | 0.501 | 0.590 | 0.843 |
| Decision Tree | 0.537 | 0.509 | 0.523 | 0.676 |

**Logistic Regression was selected as the final model**, prioritizing recall over raw accuracy — because in this business context, missing an actual churner costs more than a false alarm. Full reasoning in [`report.md`](report.md).

Key churn drivers identified: fiber-optic internet service, high total charges, and month-to-month contracts. Key protective factors: long tenure and longer-term contracts (one/two-year). Full breakdown in [`reports/feature_importance.csv`](reports/feature_importance.csv).

## Project Structure

```
customer-churn-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── CustomerChurn.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── models/
│   ├── churn_pipeline.pkl
│   └── churn_threshold.pkl
```

## How to Reproduce

### 1. Clone the repo and set up the environment
```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the dataset
Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place it in `data/` as `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

### 3. Train the model
```bash
cd src
python train.py
```
This cleans the data, trains and compares all three models, selects the best one by recall/ROC-AUC, and saves `churn_pipeline.pkl` and `churn_threshold.pkl` into `models/`. It also writes `model_comparison.csv` and `feature_importance.csv` into `reports/`.

### 4. Make a prediction
```bash
python predict.py \
  --gender Female --senior_citizen 0 --partner Yes --dependents No --tenure 5 \
  --phone_service Yes --multiple_lines No --internet_service "Fiber optic" \
  --online_security No --online_backup No --device_protection No \
  --tech_support No --streaming_tv Yes --streaming_movies Yes \
  --contract "Month-to-month" --paperless_billing Yes \
  --payment_method "Electronic check" \
  --monthly_charges 95.50 --total_charges 477.50
```
Expected output:
```
Churn probability: 88.77%
Decision threshold: 49.78%
Prediction: CHURN RISK
```

## Methodology

1. **Data cleaning** — handled blank `TotalCharges` values, dropped the non-predictive `customerID` column
2. **Preprocessing** — one-hot encoding for categorical features, standard scaling for numeric features, combined into a single `sklearn.Pipeline` to prevent data leakage between train/test splits
3. **Class imbalance** — handled via `class_weight="balanced"` on all three models (churn is ~27% of the dataset)
4. **Model comparison** — Logistic Regression, Decision Tree, and Random Forest evaluated on precision, recall, F1, and ROC-AUC, not just accuracy (accuracy is misleading on imbalanced data)
5. **Threshold tuning** — rather than using the default 0.5 cutoff, the decision threshold was tuned to target ~80% recall, reflecting the business cost tradeoff (a missed churner is costlier than a false alarm)
6. **Model selection** — Logistic Regression chosen for the best recall + ROC-AUC combination, and for being the most interpretable of the three (a relevant factor for a business stakeholder audience)

## Tech Stack
- Python, pandas, NumPy
- scikit-learn (preprocessing, modeling, evaluation)
- matplotlib / seaborn (visualization)

## Author
[Your name] — built as part of a self-directed ML learning roadmap.
