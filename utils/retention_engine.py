"""
ChurnGuard AI - Retention Strategy Engine
Generates personalised retention strategies based on customer profile and churn risk.
"""

from __future__ import annotations
from typing import Dict, List, Tuple


# ── Risk Thresholds ──────────────────────────────────────────────────────────
LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.60


def get_risk_category(probability: float) -> Tuple[str, str, str]:
    """
    Classify churn probability into risk category.

    Returns:
        (category_label, hex_color, emoji)
    """
    if probability < LOW_RISK_THRESHOLD:
        return "Low Risk", "#00d4aa", "🟢"
    elif probability < HIGH_RISK_THRESHOLD:
        return "Medium Risk", "#f59e0b", "🟡"
    else:
        return "High Risk", "#ef4444", "🔴"


def generate_retention_strategies(
    customer: Dict, churn_probability: float
) -> List[Dict]:
    """
    Generate personalised retention strategies.

    Args:
        customer: Dict of customer attributes (raw, pre-encoding)
        churn_probability: Predicted churn probability (0-1)

    Returns:
        List of strategy dicts ordered by priority
    """
    risk_cat, _, _ = get_risk_category(churn_probability)
    strategies: List[Dict] = []

    # ── High-Risk Strategies ─────────────────────────────────────────────
    if risk_cat == "High Risk":
        strategies.append(
            {
                "icon": "🎁",
                "title": "Emergency Retention Package",
                "description": (
                    "Offer an exclusive 3-month bundle: 5 % cashback on all "
                    "transactions, zero forex charges, and a free locker facility."
                ),
                "priority": "CRITICAL",
                "badge_color": "#ef4444",
                "estimated_retention": "78 %",
            }
        )
        strategies.append(
            {
                "icon": "📞",
                "title": "Dedicated Relationship Manager",
                "description": (
                    "Assign a personal banker for 24 / 7 priority support "
                    "with a guaranteed 30-minute SLA on all queries."
                ),
                "priority": "HIGH",
                "badge_color": "#f97316",
                "estimated_retention": "65 %",
            }
        )
        strategies.append(
            {
                "icon": "💳",
                "title": "Complimentary Card Upgrade",
                "description": (
                    "Upgrade to the next card tier instantly — includes "
                    "travel insurance, airport lounge access, and 3× reward points."
                ),
                "priority": "HIGH",
                "badge_color": "#f97316",
                "estimated_retention": "62 %",
            }
        )

    # ── Medium-Risk Strategies ───────────────────────────────────────────
    if risk_cat in ("Medium Risk", "High Risk"):
        strategies.append(
            {
                "icon": "💰",
                "title": "Reduced Loan Interest Offer",
                "description": (
                    "Pre-approved personal loan at 2 % below the standard rate "
                    "for the next 90 days — no processing fee."
                ),
                "priority": "MEDIUM",
                "badge_color": "#f59e0b",
                "estimated_retention": "55 %",
            }
        )
        strategies.append(
            {
                "icon": "⭐",
                "title": "Elite Loyalty Rewards Enrolment",
                "description": (
                    "Enrol in the Elite Rewards programme with a 3× points "
                    "multiplier for the first 6 months."
                ),
                "priority": "MEDIUM",
                "badge_color": "#f59e0b",
                "estimated_retention": "48 %",
            }
        )

    # ── Feature-Specific Strategies ──────────────────────────────────────
    if int(customer.get("IsActiveMember", 1)) == 0:
        strategies.append(
            {
                "icon": "📱",
                "title": "Digital Banking Activation Bonus",
                "description": (
                    "12-month complimentary subscription to the premium mobile "
                    "banking suite (ChurnGuard Pro) + ₹500 activation bonus."
                ),
                "priority": "HIGH",
                "badge_color": "#8b5cf6",
                "estimated_retention": "52 %",
            }
        )

    if int(customer.get("NumOfProducts", 2)) <= 1:
        strategies.append(
            {
                "icon": "🏦",
                "title": "Cross-Sell Bundle Offer",
                "description": (
                    "Special pricing on a savings account + micro-investment "
                    "portfolio bundle — waived maintenance fee for year one."
                ),
                "priority": "MEDIUM",
                "badge_color": "#06b6d4",
                "estimated_retention": "45 %",
            }
        )

    if float(customer.get("Balance", 100000)) < 15000:
        strategies.append(
            {
                "icon": "💎",
                "title": "High-Yield Savings Invitation",
                "description": (
                    "Exclusive invite to our 4.5 % APY savings account "
                    "with zero balance requirement and zero monthly fees."
                ),
                "priority": "MEDIUM",
                "badge_color": "#10b981",
                "estimated_retention": "42 %",
            }
        )

    if int(customer.get("OnlineBankingUsage", 1)) == 0:
        strategies.append(
            {
                "icon": "🌐",
                "title": "Digital Onboarding Campaign",
                "description": (
                    "Guided onboarding with a dedicated digital coach + "
                    "₹200 cashback on first 5 online transactions."
                ),
                "priority": "LOW",
                "badge_color": "#3b82f6",
                "estimated_retention": "38 %",
            }
        )

    # ── Low-Risk Appreciation ────────────────────────────────────────────
    if risk_cat == "Low Risk":
        strategies.append(
            {
                "icon": "🎖️",
                "title": "Customer Loyalty Recognition",
                "description": (
                    "Send a personalised thank-you gift + anniversary bonus "
                    "points to reinforce positive sentiment."
                ),
                "priority": "LOW",
                "badge_color": "#00d4aa",
                "estimated_retention": "94 %",
            }
        )

    # Deduplicate and cap
    seen_titles: set = set()
    unique: List[Dict] = []
    for s in strategies:
        if s["title"] not in seen_titles:
            seen_titles.add(s["title"])
            unique.append(s)

    return unique[:5]


def get_action_plan(churn_probability: float) -> Dict:
    """Return a short action-plan summary for a given risk level."""
    risk_cat, color, emoji = get_risk_category(churn_probability)

    plans = {
        "High Risk": {
            "immediate": "Contact customer within 24 hours via phone + email.",
            "short_term": "Deploy retention package within 72 hours.",
            "long_term": "Monthly check-in calls for next 6 months.",
            "escalation": "Flag to Senior Relationship Manager if no response in 48 h.",
        },
        "Medium Risk": {
            "immediate": "Send personalised email campaign within 48 hours.",
            "short_term": "Offer targeted product within 1 week.",
            "long_term": "Quarterly engagement review.",
            "escalation": "Escalate if balance drops or product count falls to 1.",
        },
        "Low Risk": {
            "immediate": "No urgent action required.",
            "short_term": "Include in next monthly rewards newsletter.",
            "long_term": "Annual loyalty review and tier upgrade assessment.",
            "escalation": "Monitor for sudden behavioural changes.",
        },
    }
    return {"category": risk_cat, "color": color, "emoji": emoji, **plans[risk_cat]}
