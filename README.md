# 🛡️ ChurnGuard AI
### AI-Powered Banking Churn Prediction & Customer Retention System

> **Hackathon-Ready · Production-Style · Premium UI/UX**

---

## 📌 Project Overview

**ChurnGuard AI** is a full-stack machine learning product that predicts banking customer churn, visualises risk at scale, and automatically generates personalised retention strategies — all delivered through a stunning, interactive Streamlit dashboard.

| Attribute        | Detail                                      |
|-----------------|---------------------------------------------|
| **Domain**       | FinTech · Customer Intelligence             |
| **Tech Stack**   | Python · Streamlit · Scikit-learn · Plotly  |
| **ML Models**    | LR · Decision Tree · Random Forest · XGBoost |
| **Dataset Size** | 10,000 synthetic banking customer records   |
| **UI Theme**     | Glassmorphism · Dark fintech aesthetic      |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/your-username/ChurnGuard-AI.git
cd ChurnGuard-AI
pip install -r requirements.txt
```

### 2. Train Models
```bash
python train_model.py
```
This will:
- Generate a synthetic 10,000-record banking dataset
- Train Logistic Regression, Decision Tree, Random Forest, and XGBoost
- Save the best model (by AUC-ROC) and all artifacts to `models/`

### 3. Launch Dashboard
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Demo Login Credentials:**
| Username | Password    | Role       |
|----------|-------------|------------|
| `admin`  | `admin123`  | Administrator |
| `demo`   | `demo2024`  | Analyst    |

---

## 🗂️ Project Structure

```
ChurnGuard-AI/
├── app.py                    # ← Main Streamlit dashboard
├── train_model.py            # ← ML training pipeline
├── requirements.txt
├── README.md
│
├── .streamlit/
│   └── config.toml           # Streamlit theme config
│
├── data/
│   ├── __init__.py
│   ├── generate_data.py      # Synthetic dataset generator
│   └── banking_churn_data.csv  # (created at runtime)
│
├── models/                   # (created at runtime)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── best_model_name.pkl
│   ├── training_results.pkl
│   └── *.pkl                 # Individual model files
│
└── utils/
    ├── __init__.py
    ├── data_utils.py         # Preprocessing & feature engineering
    ├── model_utils.py        # Model loading & prediction helpers
    ├── retention_engine.py   # AI retention strategy generator
    ├── segmentation.py       # Customer clustering and segment summaries
    ├── report_generator.py   # PDF prediction reports
    └── auth.py               # Demo authentication
```

---

## 🧠 ML Pipeline

```
Raw Data → Preprocessing → Feature Engineering → Train/Test Split (80/20)
    ↓
Multiple ML Models trained simultaneously
    ↓
Evaluation: Accuracy · Precision · Recall · F1 · AUC-ROC
    ↓
Best Model selected by AUC-ROC → Saved to disk
    ↓
Streamlit App loads saved model for real-time inference
```

### Feature Engineering
| Engineered Feature   | Formula                                              |
|---------------------|------------------------------------------------------|
| `BalanceToSalary`   | `Balance / (EstimatedSalary + 1)`                   |
| `ProductsPerTenure` | `NumOfProducts / (Tenure + 1)`                      |
| `EngagementScore`   | `IsActive + OnlineBanking + TxFreq/30`              |
| `CreditBand`        | `(CreditScore - 300) / 550`                         |
| `HighBalance`       | `1 if Balance > median else 0`                      |
| `SeniorCustomer`    | `1 if Age > 55 else 0`                              |

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | KPI cards, churn overview, quick analytics, feature grid |
| 🔮 **Predict** | Customer input form, churn probability gauge, risk badge, retention strategies, action plan |
| 📊 **Analytics** | 10+ interactive Plotly charts — geography, demographics, behaviour, financials |
| 🧩 **Segmentation** | K-Means segments, customer counts, and segment churn rates |
| 🏆 **Model Performance** | ROC curves, confusion matrix, metric comparison, feature importance |
| 📂 **Dataset Explorer** | Filterable data table, descriptive statistics, CSV export |
| 📋 **History** | Session-level prediction audit log with CSV export |
| 🤖 **AI Chatbot** | Natural-language assistance for churn and retention questions |

---

## 🎯 Retention Strategy Engine

Strategies are generated based on the customer's churn probability and specific attributes:

| Trigger | Strategy |
|---------|----------|
| High Risk (>60%) | Emergency retention package + dedicated RM |
| Medium Risk (30-60%) | Loyalty rewards + reduced loan interest |
| Inactive member | Digital banking activation bonus |
| Single product | Cross-sell bundle offer |
| Low balance | High-yield savings account invitation |
| No online banking | Digital onboarding campaign |
| Low Risk (<30%) | Loyalty appreciation + anniversary rewards |

---

## 📦 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| CustomerID | ID | Unique customer identifier |
| Age | Numeric | Customer age (18–78) |
| Gender | Categorical | Male / Female |
| Geography | Categorical | France / Germany / Spain |
| CreditScore | Numeric | 300–850 |
| Balance | Numeric | Account balance |
| EstimatedSalary | Numeric | Annual salary estimate |
| NumOfProducts | Numeric | 1–4 bank products |
| Tenure | Numeric | Years as customer (0–10) |
| IsActiveMember | Binary | 0/1 |
| TransactionFrequency | Numeric | Monthly transactions |
| AccountActivity | Ordinal | Low / Medium / High |
| CardType | Categorical | Silver / Gold / Platinum / Diamond |
| OnlineBankingUsage | Binary | 0/1 |
| HasLoan | Binary | 0/1 |
| **Churn** | **Target** | 0 = Retained, 1 = Churned |

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub (ensure `models/` is **not** in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → set **Main file path** to `app.py`
4. Add any secrets if needed in **Streamlit Cloud → Settings → Secrets**
5. Click **Deploy** ✅

> **Note:** Add a `packages.txt` if you need system-level dependencies.

---

## 🔮 Future Scope

- [ ] **Real-time API endpoint** using FastAPI for model serving
- [ ] **SHAP explainability** for individual prediction breakdowns
- [ ] **Live database integration** (PostgreSQL / MongoDB)
- [ ] **Email/SMS alerting** for high-risk customers
- [ ] **AutoML pipeline** with Optuna hyperparameter optimisation
- [ ] **Multi-bank multi-tenant** architecture
- [ ] **A/B testing framework** for retention strategy effectiveness
- [ ] **Deep learning model** (LSTM for sequence-based churn patterns)

---

## 📝 PPT Presentation Outline

1. **Title Slide** — ChurnGuard AI logo, tagline, team name
2. **Problem Statement** — Customer churn costs banks $5.9T annually
3. **Our Solution** — AI-powered prediction + retention in one platform
4. **Architecture** — Data → ML Pipeline → Dashboard → Strategy Engine
5. **Dataset & Features** — 10,000 records, 16 raw columns, 22 model features after encoding and engineering
6. **ML Models & Results** — Model comparison table + ROC curve
7. **Dashboard Demo** — Screenshots of each page
8. **Retention Engine** — How strategies are generated
9. **Business Impact** — ROI calculation, retention improvement %
10. **Future Scope** — API, deep learning, multi-bank deployment
11. **Team & Contact**

---

## 👨‍💻 Tech Stack

```
Python 3.10+         → Core language
Streamlit 1.28+      → Dashboard framework
Scikit-learn 1.3+    → ML models & preprocessing
XGBoost 2.0+         → Gradient boosted trees
Pandas + NumPy       → Data manipulation
Plotly 5.15+         → Interactive charts
Seaborn + Matplotlib → Static visualisations
Joblib               → Model serialisation
ReportLab            → PDF prediction reports
```

---

## 📄 License

MIT License — Free for educational and personal use.

---

<div align="center">
  <strong>🛡️ ChurnGuard AI</strong> · Built with ❤️ for innovation competitions & portfolio projects
</div>
