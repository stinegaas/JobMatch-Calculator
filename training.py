import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

data = pd.read_csv("job_chances.csv")

X = data[['punctuality','gpa','technical','mood','chemistry','luck']] 
y = data['job_offer'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40, stratify=y)

# Making scaler and scaling dataset
scaler = MinMaxScaler(feature_range=(0,1)) 
X_trainS = scaler.fit_transform(X_train) 
X_testS = scaler.transform(X_test) 

log = LogisticRegression(random_state=40)
log.fit(X_trainS, y_train)

# Performance check
proba = log.predict_proba(X_testS)[:, 1] 
pred = (proba >= 0.5).astype(int) 
print("AUC:", roc_auc_score(y_test, proba))
print("ACC:", accuracy_score(y_test, pred))

# Exporting feature order, mimMaxScale parameters and the model's coef and intercept
export = {
    "feature_order": ['punctuality','gpa','technical','mood','chemistry','luck'],
    "scaler": {
        "min_": scaler.min_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "data_min:": scaler.data_min_.tolist(),
        "data_max_": scaler.data_max_.tolist(),
        "feature_range": [0.0, 1.0]
    },
    "logreg": {
        "intercept": float(log.intercept_[0]),
        "coef": log.coef_[0].tolist() 
    },
    "meta": {
        "model": "LogisticRegression",
        "target": "job_offer"
    }
}

with open("model_export.json", "w", encoding="utf-8") as f:
    json.dump(export, f, ensure_ascii=False, indent=2)

print("Wrote model_export.json")
