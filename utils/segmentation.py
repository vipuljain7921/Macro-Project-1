"""
ChurnGuard AI – Customer Segmentation Engine
Performs K-Means clustering to segment customers into behavioural archetypes.
"""

from __future__ import annotations
import os, joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


SEGMENT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# ── Segment archetypes (assigned post-hoc by cluster analysis) ────────────────
SEGMENT_META = {
    0: {
        "name": "At-Risk Seniors",
        "icon": "⚠️",
        "color": "#ef4444",
        "description": "Older, lower-activity customers with declining balances. Highest churn risk.",
        "action": "Priority outreach with personalised retention packages.",
    },
    1: {
        "name": "High-Value Loyalists",
        "icon": "💎",
        "color": "#00d4ff",
        "description": "Long-tenure, high-balance customers with multiple products. Very low churn risk.",
        "action": "Maintain with VIP perks and annual loyalty recognition.",
    },
    2: {
        "name": "Digital Natives",
        "icon": "📱",
        "color": "#7c3aed",
        "description": "Young, online-active customers with moderate balances. Growth opportunity.",
        "action": "Upsell investment products and premium digital banking.",
    },
    3: {
        "name": "Passive Holders",
        "icon": "😴",
        "color": "#f59e0b",
        "description": "Low engagement, single product, infrequent transactions. Medium churn risk.",
        "action": "Re-engagement campaigns and product education.",
    },
    4: {
        "name": "Growing Professionals",
        "icon": "🚀",
        "color": "#10b981",
        "description": "Mid-career, active members with rising salary and moderate tenure.",
        "action": "Offer home loans, wealth management, and credit card upgrades.",
    },
}


SEG_FEATURES = [
    "Age", "CreditScore", "Balance", "EstimatedSalary",
    "NumOfProducts", "Tenure", "IsActiveMember",
    "TransactionFrequency", "OnlineBankingUsage",
]


def train_segmentation(df: pd.DataFrame, n_clusters: int = 5, random_state: int = 42):
    """Fit K-Means on SEG_FEATURES and save artefacts."""
    X = df[SEG_FEATURES].copy().fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(Xs)

    # PCA for 2D visualisation
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(Xs)

    os.makedirs(SEGMENT_DIR, exist_ok=True)
    joblib.dump(km,     os.path.join(SEGMENT_DIR, "seg_kmeans.pkl"))
    joblib.dump(scaler, os.path.join(SEGMENT_DIR, "seg_scaler.pkl"))
    joblib.dump(pca,    os.path.join(SEGMENT_DIR, "seg_pca.pkl"))
    joblib.dump(SEG_FEATURES, os.path.join(SEGMENT_DIR, "seg_features.pkl"))

    return labels, X_pca


def load_segmentation():
    km     = joblib.load(os.path.join(SEGMENT_DIR, "seg_kmeans.pkl"))
    scaler = joblib.load(os.path.join(SEGMENT_DIR, "seg_scaler.pkl"))
    pca    = joblib.load(os.path.join(SEGMENT_DIR, "seg_pca.pkl"))
    return km, scaler, pca


def segmentation_exists() -> bool:
    return os.path.exists(os.path.join(SEGMENT_DIR, "seg_kmeans.pkl"))


def predict_segment(customer: dict) -> tuple[int, str]:
    """Predict the cluster for a single customer dict."""
    km, scaler, _ = load_segmentation()
    row = pd.DataFrame([{f: customer.get(f, 0) for f in SEG_FEATURES}])
    Xs  = scaler.transform(row)
    cluster = int(km.predict(Xs)[0])
    meta = SEGMENT_META.get(cluster, SEGMENT_META[0])
    return cluster, meta


def get_segment_summary(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Build a per-cluster summary DataFrame."""
    df2 = df.copy()
    df2["Segment"] = labels
    df2["SegmentName"] = df2["Segment"].map(lambda x: SEGMENT_META.get(x, {}).get("name", f"Cluster {x}"))

    summary = df2.groupby(["Segment","SegmentName"]).agg(
        Count=("Churn","count"),
        ChurnRate=("Churn","mean"),
        AvgAge=("Age","mean"),
        AvgBalance=("Balance","mean"),
        AvgSalary=("EstimatedSalary","mean"),
        AvgProducts=("NumOfProducts","mean"),
        AvgTenure=("Tenure","mean"),
        ActivePct=("IsActiveMember","mean"),
    ).reset_index()
    return summary.round(2)
