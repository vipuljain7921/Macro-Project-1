"""
ChurnGuard AI – Model Training Pipeline
========================================
Trains Logistic Regression, Decision Tree, Random Forest (+ XGBoost if available),
evaluates all models, and saves the best-performing one along with its scaler
and feature metadata.

Usage:
    python train_model.py
"""

from __future__ import annotations
import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

# ── Optional XGBoost ─────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data.generate_data import generate_banking_dataset
from utils.data_utils import preprocess_dataframe


# ─────────────────────────────────────────────────────────────────────────────
def get_model_definitions() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42, max_iter=1000, C=0.5, solver="lbfgs",
            class_weight="balanced",
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, max_depth=7, min_samples_split=15,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1,
            max_depth=10, min_samples_split=10,
            class_weight="balanced",
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            random_state=42, eval_metric="logloss",
            verbosity=0, n_estimators=200, learning_rate=0.08,
            max_depth=6, subsample=0.8, scale_pos_weight=3,
        )
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, needs_scaling: bool,
                   scaler) -> dict:
    """Train and evaluate a single model."""
    _X_tr = scaler.transform(X_train) if needs_scaling else X_train
    _X_te = scaler.transform(X_test)  if needs_scaling else X_test

    model.fit(_X_tr, y_train)
    y_pred = model.predict(_X_te)
    y_prob = model.predict_proba(_X_te)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    return {
        "model": model,
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "y_test": y_test.tolist() if hasattr(y_test, "tolist") else list(y_test),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
        "needs_scaling": needs_scaling,
    }


def run_training(n_samples: int = 10000, save_csv: bool = True) -> dict:
    """Full training pipeline. Returns dict of per-model results."""

    # ── 1. Data ───────────────────────────────────────────────────────────
    print("  📊 Generating dataset …")
    df = generate_banking_dataset(n_samples)
    os.makedirs("data", exist_ok=True)
    if save_csv:
        df.to_csv("data/banking_churn_data.csv", index=False)
        print(f"     Saved  → data/banking_churn_data.csv  ({len(df):,} rows)")

    churn_rate = df["Churn"].mean()
    print(f"     Churn rate: {churn_rate:.2%}")

    # ── 2. Preprocessing ─────────────────────────────────────────────────
    print("\n  🔧 Preprocessing …")
    X, y, scaler, feature_names = preprocess_dataframe(df, fit_scaler=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"     Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"     Features: {len(feature_names)}")

    # ── 3. Training ───────────────────────────────────────────────────────
    print("\n  🤖 Training models …")
    model_defs = get_model_definitions()
    results: dict = {}

    scale_models = {"Logistic Regression"}

    for name, model in model_defs.items():
        print(f"     [{name}] ", end="", flush=True)
        needs_scaling = name in scale_models
        res = evaluate_model(model, X_train, X_test, y_train, y_test,
                              needs_scaling=needs_scaling, scaler=scaler)
        results[name] = res
        print(
            f"Acc={res['accuracy']:.3f}  F1={res['f1']:.3f}  AUC={res['roc_auc']:.3f}"
        )

    # ── 4. Save Artifacts ─────────────────────────────────────────────────
    print("\n  💾 Saving artifacts …")
    os.makedirs("models", exist_ok=True)

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = results[best_name]["model"]

    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler,       "models/scaler.pkl")
    joblib.dump(feature_names,"models/feature_names.pkl")
    joblib.dump(best_name,    "models/best_model_name.pkl")
    print(f"     Best model: {best_name}  (AUC={results[best_name]['roc_auc']:.4f})")

    # Save individual models
    for name, res in results.items():
        safe = name.replace(" ", "_").lower()
        joblib.dump(res["model"], f"models/{safe}.pkl")

    # Serialisable results (strip model objects)
    slim = {
        k: {kk: vv for kk, vv in v.items() if kk != "model"}
        for k, v in results.items()
    }
    slim["_feature_names"] = feature_names
    slim["_best_model_name"] = best_name

    with open("models/training_results.pkl", "wb") as f:
        pickle.dump(slim, f)

    print("     All artifacts saved to models/")

    return results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  🛡️  ChurnGuard AI  –  Model Training Pipeline")
    print("═" * 60 + "\n")

    results = run_training(n_samples=10000)

    print("\n" + "─" * 60)
    print("  📈 Final Performance Summary")
    print("─" * 60)
    header = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'AUC':>8}"
    print(header)
    print("─" * 60)
    for name, res in results.items():
        print(
            f"{name:<22} "
            f"{res['accuracy']:>9.4f} "
            f"{res['precision']:>10.4f} "
            f"{res['recall']:>8.4f} "
            f"{res['f1']:>7.4f} "
            f"{res['roc_auc']:>8.4f}"
        )
    print("─" * 60)
    print("\n  ✅  Training complete!  Run: streamlit run app.py\n")
