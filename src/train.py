import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import TARGET, build_features

ROOT = Path(__file__).resolve().parents[1]

def train(data_path=None, output_dir=None):
    data_path = Path(data_path or ROOT/"data"/"ecommerce_returns.csv")
    output_dir = Path(output_dir or ROOT/"artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain target column {TARGET!r}")
    df = df.dropna(subset=[TARGET]).copy()
    y = df[TARGET].astype(str).str.strip().str.lower().map(
        {"returned": 1, "not returned": 0, "not_returned": 0}
    )
    if y.isna().any():
        raise ValueError("Unexpected Return_Status values. Expected Returned / Not Returned.")

    X = build_features(df)
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    results = {}
    trained = {}
    for name, estimator in candidates.items():
        model = Pipeline([("preprocess", prep), ("classifier", estimator)])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob))
        }
        results[name] = metrics
        trained[name] = model

    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    joblib.dump(trained[best_name], output_dir/"best_model.joblib")
    metadata = {
        "best_model": best_name,
        "target": TARGET,
        "positive_class": "Returned",
        "features": list(X.columns),
        "metrics": results,
        "dataset_rows": int(len(df))
    }
    (output_dir/"metrics.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT/"data"/"ecommerce_returns.csv"))
    parser.add_argument("--output", default=str(ROOT/"artifacts"))
    args = parser.parse_args()
    train(args.data, args.output)
