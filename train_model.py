import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from download_data import get_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_evaluate():
    csv_path = get_data()
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Dataset shape: {df.shape}")
    print("Columns:", list(df.columns))

    # Features and Target
    X = df.drop(columns=['target'])
    y = df['target']
    feature_names = list(X.columns)

    # Train / Test Split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models definition
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42)
    }

    metrics = {}
    best_model_name = "Random Forest"
    best_recall = 0

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics[name] = {
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(prec) * 100, 2),
            "recall": round(float(rec) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "roc_auc": round(float(auc) * 100, 2),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

        print(f"\n[{name}]")
        print(f"Accuracy : {metrics[name]['accuracy']}%")
        print(f"Recall   : {metrics[name]['recall']}% (Crucial in medical diagnosis)")
        print(f"F1 Score : {metrics[name]['f1_score']}%")
        print(f"ROC-AUC  : {metrics[name]['roc_auc']}%")

        if rec > best_recall:
            best_recall = rec
            best_model_name = name

    # Feature Importance from Random Forest
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    feat_imp = [
        {"feature": feat, "importance": round(float(imp), 4)}
        for feat, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    ]

    # Save artifacts
    joblib.dump(models["Random Forest"], os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(models["Logistic Regression"], os.path.join(MODEL_DIR, "lr_model.pkl"))
    joblib.dump(models["Decision Tree"], os.path.join(MODEL_DIR, "dt_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({
            "models": metrics,
            "best_model": best_model_name,
            "feature_importance": feat_imp,
            "feature_names": feature_names,
            "dataset_summary": {
                "total_samples": len(df),
                "features_count": len(feature_names),
                "positive_cases": int(y.sum()),
                "negative_cases": int(len(y) - y.sum())
            }
        }, f, indent=4)

    print("\nAll models and evaluation metrics successfully saved to:", MODEL_DIR)

if __name__ == "__main__":
    train_and_evaluate()
