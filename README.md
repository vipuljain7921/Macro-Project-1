# ChurnGuard AI

ChurnGuard AI is a Streamlit dashboard for exploring synthetic banking customer data, estimating individual churn risk, and reviewing retention actions. It combines a trained classification model with customer segmentation and interactive portfolio summaries.

## Project Overview

Banks need a practical way to identify customers who may leave and decide where retention outreach may be useful. This project demonstrates a workflow for exploring customer churn data, checking an individual customer's risk, and reviewing suggested follow-up actions in one dashboard.

The included dataset contains 10,000 synthetic customer records. It is generated for this project and does not represent real customer accounts.

## Current Features

- **Overview:** Portfolio totals, churn and retention rates, active-member share, average balance, churn distribution, and churn rate by geography.
- **Customer risk check:** Enter customer and account details or generate a sample profile. The app displays a churn-probability gauge, risk category, customer group, suggested retention strategies, and an action plan.
- **Customer intelligence:** Search and filter records by geography, churn status, age range, card type, and customer ID; export filtered records to CSV.
- **Churn insights:** Compare churn rates by number of products and account activity.
- **Customer groups:** Review five K-means customer segments, their sizes and churn rates, and a suggested action for each group.
- **Customer insights:** View data-derived portfolio observations. An optional chat assistant can be enabled by installing the Anthropic SDK and supplying an API key in the dashboard.
- **Recent checks:** Review predictions made during the current session, view the risk-level distribution after at least three checks, and export the history as CSV or JSON.
- **Smart alerts:** Review and dismiss alerts based on churn rates and customer groups in the dataset.
- **Administrator tools:** Manage user accounts, view dataset and model-registry summaries, and change the current user's password.
- **Reports:** Download an individual prediction as JSON or PDF.

## Models and Data Processing

The training pipeline generates the synthetic dataset, preprocesses customer attributes, creates derived features, and splits the data into training and test sets. It trains Decision Tree, Random Forest, and XGBoost classifiers when XGBoost is available. Model results include standard classification metrics, and the pipeline selects the saved model by ROC-AUC.

Preprocessing encodes categorical fields, one-hot encodes geography, and derives features such as balance-to-salary ratio, products per tenure, engagement score, credit band, high-balance flag, and senior-customer flag. The segmentation module uses K-means clustering on customer and account attributes.

## Dashboard Charts

The current dashboard includes:

- A donut chart for retained versus churned customers.
- A bar chart for churn rate by geography.
- A gauge for an individual customer's predicted churn probability.
- Bar charts comparing churn rates by products held and account activity.
- A donut chart showing recent prediction risk levels after three or more checks.

## Technology Stack

- **Python 3.10 or later**
- **Streamlit** for the dashboard
- **Pandas and NumPy** for data handling
- **scikit-learn** for preprocessing, classification, and K-means clustering
- **XGBoost** for gradient-boosted classification
- **Plotly** for interactive charts
- **Joblib and pickle** for saved model and result artifacts
- **ReportLab** for downloadable PDF reports
- **Anthropic SDK (optional)** for the chat assistant

Package requirements are listed in [`requirements.txt`](requirements.txt). The optional Anthropic SDK is not included in that file.

## Project Structure

```text
ChurnGuard-AI/
├── app.py                         # Streamlit dashboard and page rendering
├── train_model.py                 # Dataset generation and model training
├── requirements.txt               # Python package requirements
├── README.md
├── .streamlit/
│   └── config.toml                # Streamlit theme and server settings
├── data/
│   ├── __init__.py
│   ├── generate_data.py           # Synthetic customer-data generator
│   └── banking_churn_data.csv     # Included synthetic dataset
├── models/
│   ├── best_model.pkl             # Selected classifier
│   ├── scaler.pkl                 # Fitted feature scaler
│   ├── feature_names.pkl          # Model input feature order
│   ├── best_model_name.pkl
│   ├── training_results.pkl       # Saved evaluation results
│   └── seg_*.pkl                  # Customer segmentation artifacts
└── utils/
    ├── __init__.py
    ├── auth.py                    # File-backed account and role helpers
    ├── data_utils.py              # Encoding and feature engineering
    ├── model_utils.py             # Model loading and prediction helpers
    ├── report_generator.py        # PDF report generation
    ├── retention_engine.py        # Risk categories and suggested actions
    └── segmentation.py            # Customer segmentation
```

## Installation

Open a terminal in the project directory and run:

```bash
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Dashboard

Start the app from the project directory:

```bash
streamlit run app.py
```

Streamlit prints a local URL in the terminal; open it in a browser. The repository includes the dataset and saved model artifacts. If the model, scaler, or feature-name artifacts are missing, the app runs the training pipeline; if the dataset is missing, it generates a replacement dataset.

To regenerate the dataset and retrain the models manually:

```bash
python train_model.py
```

New users can be registered through the sign-in screen. Accounts and roles are stored in the local `users.json` file; file-backed account data may not persist on hosting platforms with ephemeral storage.

### Optional chat assistant

To enable the Anthropic-powered assistant, install its SDK separately and enter an API key in the Customer Insights page:

```bash
python -m pip install anthropic
```

The key is held in the current Streamlit session.

## Future Scope

- Add persistent database storage for customer and account data.
- Provide an API for serving predictions to other applications.
- Add model monitoring and a documented evaluation workflow for new data.
- Explore model-explanation tools to help users understand individual predictions.
- Add automated tests for preprocessing, model loading, and dashboard workflows.

These are possible future improvements and are not features of the current application.
