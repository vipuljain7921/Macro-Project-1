"""
ChurnGuard AI - Synthetic Banking Dataset Generator
Generates realistic banking customer data for churn prediction
"""

import numpy as np
import pandas as pd


def generate_banking_dataset(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic banking customer dataset with realistic churn patterns.

    Args:
        n_samples: Number of customer records to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with customer attributes and churn labels
    """
    np.random.seed(random_state)

    # ── Customer Demographics ───────────────────────────────────────────
    customer_ids = [f"CG{str(i).zfill(7)}" for i in range(1, n_samples + 1)]
    ages = np.random.normal(40, 13, n_samples).clip(18, 78).astype(int)
    genders = np.random.choice(["Male", "Female"], n_samples, p=[0.54, 0.46])
    geographies = np.random.choice(
        ["France", "Germany", "Spain"], n_samples, p=[0.50, 0.25, 0.25]
    )

    # ── Financial Attributes ────────────────────────────────────────────
    credit_scores = np.random.normal(651, 96, n_samples).clip(300, 850).astype(int)
    # Bimodal balance: many zero balances + normal distribution of positive balances
    zero_balance_mask = np.random.random(n_samples) < 0.28
    raw_balances = np.random.normal(130000, 60000, n_samples).clip(0, 300000)
    balances = np.where(zero_balance_mask, 0.0, raw_balances)
    estimated_salaries = np.random.normal(100000, 32000, n_samples).clip(18000, 200000)
    num_products = np.random.choice([1, 2, 3, 4], n_samples, p=[0.40, 0.46, 0.10, 0.04])
    tenures = np.random.randint(0, 11, n_samples)

    # ── Behavioural / Product Attributes ───────────────────────────────
    active_members = np.random.choice([0, 1], n_samples, p=[0.38, 0.62])
    transaction_frequencies = np.random.randint(1, 31, n_samples)
    account_activities = np.random.choice(
        ["High", "Medium", "Low"], n_samples, p=[0.28, 0.42, 0.30]
    )
    card_types = np.random.choice(
        ["Silver", "Gold", "Platinum", "Diamond"], n_samples, p=[0.38, 0.32, 0.20, 0.10]
    )
    online_banking = np.random.choice([0, 1], n_samples, p=[0.33, 0.67])
    has_loan = np.random.choice([0, 1], n_samples, p=[0.60, 0.40])

    # ── Churn Label Generation (realistic probability model) ────────────
    churn_proba = (
        0.08
        + (ages > 55).astype(float) * 0.10
        + (ages > 65).astype(float) * 0.08
        + (np.array(list(map(lambda g: g == "Germany", geographies)))).astype(float) * 0.09
        + (num_products >= 3).astype(float) * 0.18
        + (num_products == 1).astype(float) * 0.05
        + (active_members == 0).astype(float) * 0.13
        + (balances == 0).astype(float) * 0.06
        + (credit_scores < 500).astype(float) * 0.09
        + (credit_scores < 400).astype(float) * 0.08
        + (transaction_frequencies < 5).astype(float) * 0.07
        + (online_banking == 0).astype(float) * 0.05
        + (tenures < 2).astype(float) * 0.04
        + np.random.normal(0, 0.04, n_samples)
    ).clip(0.01, 0.97)

    churned = (np.random.random(n_samples) < churn_proba).astype(int)

    df = pd.DataFrame(
        {
            "CustomerID": customer_ids,
            "Age": ages,
            "Gender": genders,
            "Geography": geographies,
            "CreditScore": credit_scores,
            "Balance": np.round(balances, 2),
            "EstimatedSalary": np.round(estimated_salaries, 2),
            "NumOfProducts": num_products,
            "Tenure": tenures,
            "IsActiveMember": active_members,
            "TransactionFrequency": transaction_frequencies,
            "AccountActivity": account_activities,
            "CardType": card_types,
            "OnlineBankingUsage": online_banking,
            "HasLoan": has_loan,
            "Churn": churned,
        }
    )

    return df


if __name__ == "__main__":
    df = generate_banking_dataset(10000)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Churn rate: {df['Churn'].mean():.2%}")
