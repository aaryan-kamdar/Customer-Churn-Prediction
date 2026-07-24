import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import joblib
from sklearn.pipeline import Pipeline # type: ignore
# 1. loading the pipeline  and threshold 

pipeline = joblib.load("churn_pipeline.pkl")
threshold = joblib.load("churn_threshold.pkl")
 
# Parse CLI arguments
parser = argparse.ArgumentParser(
    description="Predict customer churn using the trained Logistic Regression pipeline."
)
parser.add_argument("--gender", type=str, required=True, choices=["Male", "Female"])
parser.add_argument("--senior_citizen", type=int, required=True, choices=[0, 1])
parser.add_argument("--partner", type=str, required=True, choices=["Yes", "No"])
parser.add_argument("--dependents", type=str, required=True, choices=["Yes", "No"])
parser.add_argument("--tenure", type=int, required=True)
 
parser.add_argument("--phone_service", type=str, required=True, choices=["Yes", "No"])
parser.add_argument("--multiple_lines", type=str, required=True,
                     choices=["Yes", "No", "No phone service"])
parser.add_argument("--internet_service", type=str, required=True,
                     choices=["DSL", "Fiber optic", "No"])
 
parser.add_argument("--online_security", type=str, required=True,
                     choices=["Yes", "No", "No internet service"])
parser.add_argument("--online_backup", type=str, required=True,
                     choices=["Yes", "No", "No internet service"])
parser.add_argument("--device_protection", type=str, required=True,
                     choices=["Yes", "No", "No internet service"])
parser.add_argument("--tech_support", type=str, required=True,
                     choices=["Yes", "No", "No internet service"])
parser.add_argument("--streaming_tv", type=str, required=True,
                     choices=["Yes", "No", "No internet service"])
parser.add_argument("--streaming_movies", type=str, required=True,
                     choices=["Yes", "No", "No internet service"])
 
parser.add_argument("--contract", type=str, required=True,
                     choices=["Month-to-month", "One year", "Two year"])
parser.add_argument("--paperless_billing", type=str, required=True, choices=["Yes", "No"])
parser.add_argument("--payment_method", type=str, required=True,
                     choices=["Electronic check", "Mailed check",
                              "Bank transfer (automatic)", "Credit card (automatic)"])
 
parser.add_argument("--monthly_charges", type=float, required=True)
parser.add_argument("--total_charges", type=float, required=True)
 
args = parser.parse_args()
## creating a dataframe out of arguments
data = pd.DataFrame({
    "gender": [args.gender],
    "SeniorCitizen": [args.senior_citizen],
    "Partner": [args.partner],
    "Dependents": [args.dependents],
    "tenure": [args.tenure],
    "PhoneService": [args.phone_service],
    "MultipleLines": [args.multiple_lines],
    "InternetService": [args.internet_service],
    "OnlineSecurity": [args.online_security],
    "OnlineBackup": [args.online_backup],
    "DeviceProtection": [args.device_protection],
    "TechSupport": [args.tech_support],
    "StreamingTV": [args.streaming_tv],
    "StreamingMovies": [args.streaming_movies],
    "Contract": [args.contract],
    "PaperlessBilling": [args.paperless_billing],
    "PaymentMethod": [args.payment_method],
    "MonthlyCharges": [args.monthly_charges],
    "TotalCharges": [args.total_charges],
})

## 4. predicting 

probs=pipeline.predict_proba(data)[:,1][0]
will_churn=bool(probs>threshold)

print(f"Churn probability {probs:0.2%}")
print(f"Descision Threshold  {threshold:0.2%}")
print(f"Churn risk {will_churn}")