"""
ChurnGuard AI - Model Utilities
Handles model loading, prediction, and performance metric helpers.
"""

from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, Any


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def models_exist() -> bool:
    """Check whether trained models are present on disk."""
    required = ["best_model.pkl", "scaler.pkl", "feature_names.pkl"]
    return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required)


def load_artifacts() -> Tuple[Any, Any, list, str]:
    """
    Load the best model, scaler, feature names, and model name from disk.

    Returns:
        (model, scaler, feature_names, best_model_name)
    """
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    best_name = joblib.load(os.path.join(MODEL_DIR, "best_model_name.pkl"))
    return model, scaler, feature_names, best_name


def load_training_results() -> Dict | None:
    """Load saved training results (metrics for all models)."""
    path = os.path.join(MODEL_DIR, "training_results.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_churn(model, X: np.ndarray) -> Tuple[int, float]:
    """
    Run churn prediction.

    Returns:
        (prediction_label, churn_probability)
    """
    prob = model.predict_proba(X)[0][1]
    label = int(prob >= 0.5)
    return label, float(prob)


def get_feature_importances(model, feature_names: list) -> pd.DataFrame | None:
    """
    Extract feature importances if the model supports them.

    Returns:
        DataFrame with 'feature' and 'importance' columns, sorted descending
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def generate_sample_customer() -> Dict:
    """Return a randomly generated sample customer for demo purposes."""
    np.random.seed(None)  # Random each call
    return {
        "Age": int(np.random.randint(22, 70)),
        "Gender": np.random.choice(["Male", "Female"]),
        "Geography": np.random.choice(["France", "Germany", "Spain"]),
        "CreditScore": int(np.random.randint(350, 850)),
        "Balance": float(round(np.random.uniform(0, 250000), 2)),
        "EstimatedSalary": float(round(np.random.uniform(25000, 180000), 2)),
        "NumOfProducts": int(np.random.choice([1, 2, 3, 4])),
        "Tenure": int(np.random.randint(0, 11)),
        "IsActiveMember": int(np.random.choice([0, 1])),
        "TransactionFrequency": int(np.random.randint(1, 31)),
        "AccountActivity": np.random.choice(["High", "Medium", "Low"]),
        "CardType": np.random.choice(["Silver", "Gold", "Platinum", "Diamond"]),
        "OnlineBankingUsage": int(np.random.choice([0, 1])),
        "HasLoan": int(np.random.choice([0, 1])),
    }
