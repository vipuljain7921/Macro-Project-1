"""
ChurnGuard AI - Data Utilities
Handles preprocessing, feature engineering, and encoding.
All mappings are deterministic (no fitted encoders needed for categoricals).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ── Deterministic Encoding Maps ─────────────────────────────────────────────
GENDER_MAP = {"Male": 1, "Female": 0}
ACTIVITY_MAP = {"Low": 0, "Medium": 1, "High": 2}
CARD_MAP = {"Silver": 0, "Gold": 1, "Platinum": 2, "Diamond": 3}
GEOGRAPHIES = ["France", "Germany", "Spain"]


def preprocess_dataframe(df: pd.DataFrame, scaler: StandardScaler | None = None,
                          fit_scaler: bool = False):
    """
    Full preprocessing pipeline for a DataFrame (batch / training use).

    Args:
        df: Raw dataframe (may contain CustomerID and Churn columns)
        scaler: Pre-fitted StandardScaler; if None and fit_scaler=True, a new one is fitted
        fit_scaler: Whether to fit a new scaler on this data

    Returns:
        X (features array), y (labels or None), scaler, feature_column_names
    """
    df = df.copy()

    # Drop identifier
    if "CustomerID" in df.columns:
        df.drop("CustomerID", axis=1, inplace=True)

    # Separate target
    y = None
    if "Churn" in df.columns:
        y = df.pop("Churn").values

    # ── Encode Categoricals ──────────────────────────────────────────────
    df["Gender"] = df["Gender"].map(GENDER_MAP).fillna(0).astype(int)
    df["AccountActivity"] = df["AccountActivity"].map(ACTIVITY_MAP).fillna(1).astype(int)
    df["CardType"] = df["CardType"].map(CARD_MAP).fillna(0).astype(int)

    # One-hot encode Geography with fixed column order
    for geo in GEOGRAPHIES:
        df[f"Geography_{geo}"] = (df["Geography"] == geo).astype(int)
    df.drop("Geography", axis=1, inplace=True)

    # ── Feature Engineering ──────────────────────────────────────────────
    df["BalanceToSalary"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["ProductsPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["EngagementScore"] = (
        df["IsActiveMember"] + df["OnlineBankingUsage"] + df["TransactionFrequency"] / 30
    )
    df["CreditBand"] = (df["CreditScore"] - 300) / 550  # normalised 0-1
    df["HighBalance"] = (df["Balance"] > df["Balance"].median()).astype(int)
    df["SeniorCustomer"] = (df["Age"] > 55).astype(int)

    feature_names = df.columns.tolist()

    # ── Scaling ──────────────────────────────────────────────────────────
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(df)
    elif scaler is not None:
        X = scaler.transform(df)
    else:
        X = df.values

    return X, y, scaler, feature_names


def preprocess_single_customer(customer: dict, scaler: StandardScaler,
                                 feature_names: list) -> np.ndarray:
    """
    Transform a single customer dict into the scaled feature vector expected by the model.

    Args:
        customer: Dict with raw customer attributes
        scaler: Fitted StandardScaler from training
        feature_names: Ordered list of feature column names from training

    Returns:
        Numpy array of shape (1, n_features)
    """
    df = pd.DataFrame([customer])

    # Apply same encoding
    df["Gender"] = df["Gender"].map(GENDER_MAP).fillna(0).astype(int)
    df["AccountActivity"] = df["AccountActivity"].map(ACTIVITY_MAP).fillna(1).astype(int)
    df["CardType"] = df["CardType"].map(CARD_MAP).fillna(0).astype(int)

    for geo in GEOGRAPHIES:
        df[f"Geography_{geo}"] = (df["Geography"] == geo).astype(int)
    if "Geography" in df.columns:
        df.drop("Geography", axis=1, inplace=True)

    # Feature engineering
    df["BalanceToSalary"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["ProductsPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["EngagementScore"] = (
        df["IsActiveMember"] + df["OnlineBankingUsage"] + df["TransactionFrequency"] / 30
    )
    df["CreditBand"] = (df["CreditScore"] - 300) / 550
    df["HighBalance"] = (df["Balance"] > 100000).astype(int)
    df["SeniorCustomer"] = (df["Age"] > 55).astype(int)

    # Align columns to training feature order; fill missing with 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    return scaler.transform(df)


def get_feature_display_names() -> dict:
    """Human-readable feature names for display in charts."""
    return {
        "Age": "Customer Age",
        "Gender": "Gender",
        "CreditScore": "Credit Score",
        "Balance": "Account Balance",
        "EstimatedSalary": "Estimated Salary",
        "NumOfProducts": "Number of Products",
        "Tenure": "Tenure (Years)",
        "IsActiveMember": "Is Active Member",
        "TransactionFrequency": "Transaction Frequency",
        "AccountActivity": "Account Activity Level",
        "CardType": "Card Type",
        "OnlineBankingUsage": "Online Banking Usage",
        "HasLoan": "Has Active Loan",
        "Geography_France": "Located in France",
        "Geography_Germany": "Located in Germany",
        "Geography_Spain": "Located in Spain",
        "BalanceToSalary": "Balance-to-Salary Ratio",
        "ProductsPerTenure": "Products per Year",
        "EngagementScore": "Engagement Score",
        "CreditBand": "Credit Band (Normalised)",
        "HighBalance": "High Balance Flag",
        "SeniorCustomer": "Senior Customer Flag",
    }
