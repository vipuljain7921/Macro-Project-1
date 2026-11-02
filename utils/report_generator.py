"""
ChurnGuard AI – PDF Report Generator
Generates a professional single-page prediction report using ReportLab.
"""

from __future__ import annotations
import io, datetime, os
from typing import List, Dict

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, Image as ReportImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

SHIELD_ICON = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assets", "churnguard-shield.png"
)

# ── Brand colours ─────────────────────────────────────────────────────────────
C_DARK   = colors.HexColor("#0a0f1e")
C_MID    = colors.HexColor("#0e1628")
C_ACCENT = colors.HexColor("#00d4ff")
C_PURPLE = colors.HexColor("#7c3aed")
C_GREEN  = colors.HexColor("#10b981")
C_YELLOW = colors.HexColor("#f59e0b")
C_RED    = colors.HexColor("#ef4444")
C_TEXT   = colors.HexColor("#1e293b")
C_MUTED  = colors.HexColor("#64748b")
C_WHITE  = colors.white

RISK_COLORS = {"Low Risk": C_GREEN, "Medium Risk": C_YELLOW, "High Risk": C_RED}
PRIORITY_COLORS = {
    "CRITICAL": C_RED, "HIGH": colors.HexColor("#f97316"),
    "MEDIUM": C_YELLOW, "LOW": C_GREEN,
}


# ── Style helpers ─────────────────────────────────────────────────────────────
def _styles():
    base = getSampleStyleSheet()
    extra = {
        "Title": ParagraphStyle("CGTitle", fontSize=22, fontName="Helvetica-Bold",
                                 textColor=C_WHITE, alignment=TA_CENTER, leading=28),
        "SubTitle": ParagraphStyle("CGSub", fontSize=10, fontName="Helvetica",
                                    textColor=colors.HexColor("#b0bfd8"), alignment=TA_CENTER, leading=14),
        "SectionHead": ParagraphStyle("CGSec", fontSize=11, fontName="Helvetica-Bold",
                                       textColor=C_ACCENT, leading=16, spaceBefore=8, spaceAfter=4),
        "Body": ParagraphStyle("CGBody", fontSize=9, fontName="Helvetica",
                                textColor=C_TEXT, leading=13),
        "BodyMuted": ParagraphStyle("CGMuted", fontSize=8, fontName="Helvetica",
                                     textColor=C_MUTED, leading=12),
        "MetricLabel": ParagraphStyle("CGMLbl", fontSize=7.5, fontName="Helvetica-Bold",
                                       textColor=C_MUTED, alignment=TA_CENTER,
                                       textTransform="uppercase"),
        "MetricValue": ParagraphStyle("CGMVal", fontSize=16, fontName="Helvetica-Bold",
                                       textColor=C_ACCENT, alignment=TA_CENTER, leading=20),
        "RiskText": ParagraphStyle("CGRisk", fontSize=13, fontName="Helvetica-Bold",
                                    alignment=TA_CENTER),
        "Small": ParagraphStyle("CGSmall", fontSize=7.5, fontName="Helvetica",
                                  textColor=C_MUTED, leading=11),
        "Footer": ParagraphStyle("CGFoot", fontSize=7.5, fontName="Helvetica",
                                  textColor=C_MUTED, alignment=TA_CENTER),
    }
    return {**{k: base[k] for k in base.byName}, **extra}


# ── Public API ────────────────────────────────────────────────────────────────
def generate_pdf_report(
    customer: Dict,
    churn_probability: float,
    risk_category: str,
    strategies: List[Dict],
    action_plan: Dict,
    model_name: str = "Best Model",
) -> bytes:
    """
    Generate a complete PDF prediction report.

    Returns:
        Raw PDF bytes (suitable for st.download_button)
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="ChurnGuard AI Prediction Report",
        author="ChurnGuard AI",
    )

    S = _styles()
    W = A4[0] - 32 * mm   # usable width
    story = []

    # ── Header Banner ────────────────────────────────────────────────────
    header_data = [[
        ReportImage(SHIELD_ICON, width=14 * mm, height=14 * mm * 672 / 568),
        Paragraph("ChurnGuard AI", S["Title"]),
    ]]
    header_tbl = Table(header_data, colWidths=[20 * mm, W - 20 * mm])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_DARK),
        ("ROWPADDING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROUNDEDCORNERS", [8]),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("AI-Powered Banking Churn Prediction Report", S["SubTitle"]))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}  ·  Model: {model_name}",
        S["SubTitle"],
    ))
    story.append(Spacer(1, 5 * mm))
    story.append(HRFlowable(width=W, color=C_ACCENT, thickness=1.5))
    story.append(Spacer(1, 4 * mm))

    # ── Churn Probability + Risk ─────────────────────────────────────────
    risk_color = RISK_COLORS.get(risk_category, C_YELLOW)
    pct_text = f"{churn_probability:.1%}"

    kpi_data = [
        [
            _metric_cell(pct_text, "Churn Probability", S, C_RED if churn_probability > 0.6 else C_YELLOW if churn_probability > 0.3 else C_GREEN),
            _metric_cell(risk_category, "Risk Category", S, risk_color),
            _metric_cell("YES" if churn_probability >= 0.5 else "NO", "Predicted Churn", S,
                         C_RED if churn_probability >= 0.5 else C_GREEN),
            _metric_cell(f"{1 - churn_probability:.1%}", "Retention Score", S, C_GREEN),
        ]
    ]
    kpi_tbl = Table(kpi_data, colWidths=[W / 4] * 4)
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWPADDING", (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 5 * mm))

    # ── Customer Profile ─────────────────────────────────────────────────
    story.append(Paragraph("Customer Profile", S["SectionHead"]))

    profile_items = [
        ("Age", str(customer.get("Age", "–"))),
        ("Gender", str(customer.get("Gender", "–"))),
        ("Geography", str(customer.get("Geography", "–"))),
        ("Credit Score", str(customer.get("CreditScore", "–"))),
        ("Account Balance", f"${float(customer.get('Balance', 0)):,.2f}"),
        ("Estimated Salary", f"${float(customer.get('EstimatedSalary', 0)):,.2f}"),
        ("Number of Products", str(customer.get("NumOfProducts", "–"))),
        ("Tenure (Years)", str(customer.get("Tenure", "–"))),
        ("Active Member", "Yes" if customer.get("IsActiveMember", 0) else "No"),
        ("Tx Frequency / Month", str(customer.get("TransactionFrequency", "–"))),
        ("Account Activity", str(customer.get("AccountActivity", "–"))),
        ("Card Type", str(customer.get("CardType", "–"))),
        ("Online Banking", "Yes" if customer.get("OnlineBankingUsage", 0) else "No"),
        ("Has Active Loan", "Yes" if customer.get("HasLoan", 0) else "No"),
    ]

    # Two-column grid
    rows = []
    for i in range(0, len(profile_items), 2):
        left  = profile_items[i]
        right = profile_items[i + 1] if i + 1 < len(profile_items) else ("", "")
        rows.append([
            Paragraph(left[0], S["BodyMuted"]),  Paragraph(left[1], S["Body"]),
            Paragraph(right[0], S["BodyMuted"]), Paragraph(right[1], S["Body"]),
        ])

    col_w = [W * 0.18, W * 0.32, W * 0.18, W * 0.32]
    profile_tbl = Table(rows, colWidths=col_w)
    profile_tbl.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#e2e8f0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(profile_tbl)
    story.append(Spacer(1, 5 * mm))

    # ── Retention Strategies ─────────────────────────────────────────────
    story.append(Paragraph("AI-Generated Retention Strategies", S["SectionHead"]))

    for s in strategies[:5]:
        p_color = PRIORITY_COLORS.get(s.get("priority", "LOW"), C_MUTED)
        strat_data = [[
            Paragraph(f"{s.get('icon','•')} {s.get('title','')}", S["Body"]),
            Paragraph(s.get("priority", ""), ParagraphStyle(
                "Prio", fontSize=7, fontName="Helvetica-Bold",
                textColor=p_color, alignment=TA_RIGHT)),
        ], [
            Paragraph(s.get("description", ""), S["BodyMuted"]),
            Paragraph(f"Est. retention: {s.get('estimated_retention','–')}", S["Small"]),
        ]]
        strat_tbl = Table(strat_data, colWidths=[W * 0.78, W * 0.22])
        strat_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#e2e8f0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEABOVE", (0, 0), (-1, 0), 2.5, p_color),
            ("ROUNDEDCORNERS", [4]),
        ]))
        story.append(KeepTogether([strat_tbl, Spacer(1, 2 * mm)]))

    story.append(Spacer(1, 3 * mm))

    # ── Action Plan ──────────────────────────────────────────────────────
    story.append(Paragraph("Recommended Action Plan", S["SectionHead"]))

    ap_rows = [
        [Paragraph("⚡ Immediate", S["Body"]),   Paragraph(action_plan.get("immediate", "–"), S["BodyMuted"])],
        [Paragraph("📅 Short-Term", S["Body"]),  Paragraph(action_plan.get("short_term", "–"), S["BodyMuted"])],
        [Paragraph("🔭 Long-Term", S["Body"]),   Paragraph(action_plan.get("long_term", "–"), S["BodyMuted"])],
        [Paragraph("🚨 Escalation", S["Body"]),  Paragraph(action_plan.get("escalation", "–"), S["BodyMuted"])],
    ]
    ap_tbl = Table(ap_rows, colWidths=[W * 0.22, W * 0.78])
    ap_tbl.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#e2e8f0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(ap_tbl)
    story.append(Spacer(1, 6 * mm))

    # ── Footer ───────────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, color=colors.HexColor("#e2e8f0"), thickness=0.8))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "ChurnGuard AI  ·  AI-Powered Banking Intelligence Platform  ·  "
        "For authorised internal use only  ·  © 2024",
        S["Footer"],
    ))

    doc.build(story)
    return buf.getvalue()


def _metric_cell(value: str, label: str, S: dict, value_color) -> list:
    """Helper: returns a vertical [value, label] cell content."""
    return [
        Paragraph(value, ParagraphStyle(
            "MV", fontSize=15, fontName="Helvetica-Bold",
            textColor=value_color, alignment=TA_CENTER, leading=20)),
        Paragraph(label, S["MetricLabel"]),
    ]
