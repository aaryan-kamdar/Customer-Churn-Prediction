# 1 Importing Libraries # importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.base import clone
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score,
    precision_recall_curve
)
import joblib

#2 Importing the dataset 
df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

#2.1 Dropping the customerId column
df=df.drop("customerID",axis=1)

## 3. Data cleaning converting total charges into int

df["TotalCharges"]=df["TotalCharges"].replace(" ",np.nan)
df = df.dropna(subset=["TotalCharges"], axis=0)   
df["TotalCharges"]=df["TotalCharges"].astype(float)

# 3.1 split the data into features and targets and mapping the yes as 1 and no to 0 in target
x=df.drop("Churn",axis=1)
y=df["Churn"].map({"Yes":1,"No":0})

# 3.2 Splitting the data into train and test 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# 3.3 Encoding the categorical variables 
categorical_columns=["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
categoric_encoder=OneHotEncoder(handle_unknown="ignore",drop="first")

#3.4 Scaling the numeric variables
numerical_columns=["tenure","MonthlyCharges","TotalCharges"]
numeric_encoder=StandardScaler()
 
# 3.5 Using a column transformer to make sure both the transformation i.e., scaling and one hot encoding on categrocial variable happen on one go
preprocessor=ColumnTransformer([("one_hot",categoric_encoder,categorical_columns),
                                ("scaling",numeric_encoder,numerical_columns)],
                              remainder="passthrough")

# 4.1 Modelling getting the results of three models logistic regression, Random forest classifier, decision tree classifier

lr_model=LogisticRegression(class_weight="balanced",solver="lbfgs",max_iter=1000)
rf_model=RandomForestClassifier(class_weight="balanced",n_estimators=1000,n_jobs=-1)
dt_clf_model=DecisionTreeClassifier(class_weight="balanced",criterion="entropy")

models={"Logistic Regression": lr_model,"Random forest classifier":rf_model,"Decision tree classifier":dt_clf_model}
models

# 4.2 Creating a pipeline and fitting them according to these models 
base_pipeline=Pipeline([("preprocessor",preprocessor),
                        ("model",rf_model)])

fitted_pipeline={}
for name,model in models.items():
    pipe=clone(base_pipeline)
    pipe.set_params(model=model)
    pipe.fit(x_train,y_train)
    fitted_pipeline[name]=pipe

#5 Evaluting the models

results={}
for name,pipe in fitted_pipeline.items():
    preds=pipe.predict(x_test)
    probs=pipe.predict_proba(x_test)[:, 1]

    results[name] = {
        "preds": preds,
        "probs": probs,
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }

    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
    print("ROC-AUC:", round(results[name]["roc_auc"], 4))

# 6 Build the comparision table and select the best model

comparison_df=pd.DataFrame({
    name:{
        "Precision":res["precision"],
        "Recall":res["recall"],
        "F1":res["f1"],
        "ROC-AUC":res["roc_auc"]
    }
    for name,res in results.items()
}).T


print("\n===== Model Comparison =====")
print(comparison_df.round(3))
# Recall matters the most in evaluation as you want to catch people who are tending to leave so select recall
best_model_name = comparison_df.sort_values(
    by=["Recall", "ROC-AUC"], ascending=False
).index[0]

print(f"\nSelected best model: {best_model_name}")


# 7. Building the final pipeline with the best model

final_model = models[best_model_name]   

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", final_model)
])

final_pipeline.fit(x_train, y_train)

# 8. FEATURE IMPORTANCE (coefficients) — finding the most importand features that affect the churn rate and preventing 
feature_names = final_pipeline.named_steps["preprocessor"].get_feature_names_out()
coefs = final_pipeline.named_steps["model"].coef_[0]
 
coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs}) \
    .sort_values("coefficient", ascending=False)
 
print("\nTop churn-driving features:")
print(coef_df.head(10).to_string(index=False))

print("\nTop churn-preventing features:")
print(coef_df.tail(10).to_string(index=False))

# 9. Finding the Recall-optimized threshold for final mdoel

final_probs = final_pipeline.predict_proba(x_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, final_probs)

target_recall = 0.80
idx = np.argmin(np.abs(recalls - target_recall))
chosen_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5

final_preds = (final_probs >= chosen_threshold).astype(int)

print(f"\n===== Final Model ({best_model_name}) at threshold {chosen_threshold:.4f} =====")
print(classification_report(y_test, final_preds, target_names=["No Churn", "Churn"]))
print("ROC-AUC:", round(roc_auc_score(y_test, final_probs), 4))


# 10. Saving the model

joblib.dump(final_pipeline,"churn_pipeline.pkl")
joblib.dump(chosen_threshold, "churn_threshold.pkl")