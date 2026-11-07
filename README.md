# 📡 Intelligent Customer Behaviour Analysis & Churn Prediction System
### End-to-End Data Science Portfolio Project | Japanese Telecommunications Case Study

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

---

## 📌 1. Project Overview

Customer churn represents an existential challenge for subscription and telecommunications businesses. In highly competitive and saturated markets like **Japan**, where customer acquisition cost (CAC) is 5 to 7 times higher than retention, identifying departing subscribers *before* they leave is critical to maintaining recurring profitability.

This repository implements a production-grade **Customer Intelligence & Churn Prediction System** capable of:
1. **Predictive Analytics:** Forecasting customer churn probability with a tuned **Random Forest** pipeline achieving **0.8435 ROC-AUC** and **77.54% Recall** on held-out test data.
2. **Customer Segmentation:** Clustering subscribers into 4 actionable behavioral segments using **K-Means**, identifying an acute vulnerability group with a **49.2%** churn rate.
3. **Prescriptive Analytics:** Delivering automated, actionable retention playbooks customized for Japanese telecommunications (e.g., d-point/Ponta point loyalty incentives, Hikari optical fiber bundle rebates, and automated onboarding interventions).
4. **Interactive Web Dashboard:** A 4-page **Streamlit** application enabling business executives and customer success agents to explore insights and score individual or batch customers in real time.

> **Constraint Compliance:** In strict adherence to project requirements, **no XGBoost or SHAP** has been used. The models rely on interpretable, pure Scikit-learn pipelines with feature importance decomposition and cross-validation.

---

## 🏗️ 2. System Architecture

```text
                                  ┌──────────────────────────┐
                                  │ IBM Telco Churn Dataset  │
                                  └─────────────┬────────────┘
                                                │
                                                ▼
                                  ┌──────────────────────────┐
                                  │ Data Cleaning & Pipeline │
                                  │  (Leakage-Free Encoders) │
                                  └─────────────┬────────────┘
                                                │
                    ┌───────────────────────────┴───────────────────────────┐
                    ▼                                                       ▼
      ┌───────────────────────────┐                           ┌───────────────────────────┐
      │     SQL Analytics Engine  │                           │   Feature Engineering &   │
      │   (Cohorts, Revenue Loss) │                           │    Statistical Testing    │
      └───────────────────────────┘                           └─────────────┬─────────────┘
                                                                            │
                                                ┌───────────────────────────┴───────────────────────────┐
                                                ▼                                                       ▼
                                  ┌───────────────────────────┐                           ┌───────────────────────────┐
                                  │ Supervised Classification │                           │   Unsupervised K-Means    │
                                  │ (LR, DT, Random Forest)   │                           │   Customer Segmentation   │
                                  └─────────────┬─────────────┘                           └─────────────┬─────────────┘
                                                │                                                       │
                                                └───────────────────────────┬───────────────────────────┘
                                                                            │
                                                                            ▼
                                                              ┌───────────────────────────┐
                                                              │   Streamlit Web Platform  │
                                                              │ (Overview, EDA, Predict)  │
                                                              └───────────────────────────┘
```

---

## 📂 3. Repository Structure

```text
Customer_Churn_Prediction/
│
├── dataset/
│   ├── customer_churn.csv              # IBM Telco Customer Churn dataset
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── telco_churn.db                  # SQLite database for SQL analytics
│
├── notebooks/
│   ├── 01_data_collection.ipynb        # Dataset loading, profiling, inspection
│   ├── 02_data_preprocessing.ipynb     # Missing values, target encoding, class balance
│   ├── 03_exploratory_analysis.ipynb   # Visualizing contract, tenure, charge patterns
│   ├── 04_statistical_analysis.ipynb   # Chi-Square tests and Welch's t-tests
│   ├── 05_feature_engineering.ipynb    # Domain feature construction & pipelines
│   ├── 06_model_training.ipynb         # Training LR, DT, Random Forest, KNN
│   └── 07_model_evaluation.ipynb       # Confusion matrices, ROC curves, K-Means
│
├── models/
│   ├── churn_model.pkl                 # Best production model (Random Forest Pipeline)
│   ├── clustering_model.pkl            # K-Means pipeline (StandardScaler + KMeans)
│   ├── model_comparison.csv            # Benchmark metrics across all models
│   ├── feature_importance.csv          # Random Forest feature importance ranking
│   ├── segment_summary.csv             # K-Means cluster profiles
│   └── evaluation_results.json         # Full evaluation metrics & ROC coordinates
│
├── sql/
│   └── customer_analysis.sql           # 8 enterprise SQL business queries
│
├── src/
│   ├── preprocessing.py                # Data loading, cleaning, feature pipelines
│   ├── prediction.py                   # Single & batch inference and retention engine
│   └── evaluation.py                   # Scoring metrics, cross-validation, comparisons
│
├── dashboard/
│   └── app.py                          # 4-page interactive Streamlit web application
│
├── reports/
│   └── project_report.md               # Detailed technical & business project report
│
├── create_sqlite_db.py                 # SQLite database initialization script
├── run_sql_analysis.py                 # SQL query test runner
├── train_models.py                     # End-to-end ML training & evaluation script
├── generate_notebooks.py               # Jupyter Notebook generator
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation & interview guide
```

---

## ⚡ 4. Quick Start & Execution

### Prerequisites
- Python 3.10+
- Git

### 1. Clone & Setup Environment
```bash
git clone https://github.com/HarshiniKC03/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
pip install -r requirements.txt
```

### 2. Run Database & SQL Analysis
```bash
python create_sqlite_db.py
python run_sql_analysis.py
```

### 3. Train Models & Generate Benchmarks
```bash
python train_models.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501` to view the interactive application.

---

## 📊 5. Machine Learning Benchmark Results

All four candidate algorithms were evaluated on the same 20% holdout test set (1,409 customers) using Stratified 5-Fold Cross Validation:

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score | ROC-AUC | True Positives | False Negatives | False Positives | True Negatives |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** ⭐ | **76.30%** | **53.70%** | **77.54%** | **0.6346** | **0.8435** | 290 | 84 | 250 | 785 |
| **Decision Tree** | 74.24% | 50.94% | 79.41% | 0.6207 | 0.8344 | 297 | 77 | 286 | 749 |
| **Logistic Regression** | 73.95% | 50.60% | 78.88% | 0.6165 | 0.8415 | 295 | 79 | 288 | 747 |
| **K-Nearest Neighbors** | 75.80% | 54.65% | 51.87% | 0.5322 | 0.7871 | 194 | 180 | 161 | 874 |

### Why Random Forest is the Selected Production Model:
1. **Best Discrimination:** Highest ROC-AUC score (**0.8435**).
2. **Cost-Effective Precision-Recall Balance:** Captures **77.54% of churners** while triggering **36 fewer False Positives** than Logistic Regression, avoiding wasted marketing expenditure.
3. **Nonlinear Interaction Capture:** Accurately models interactions between fiber optic broadband, month-to-month contracts, and technical support deficits without requiring manual polynomial expansion.

---

## 👥 6. Customer Segmentation (K-Means Clustering)

Subscribers are clustered into 4 distinct behavioral segments using standardized `tenure` and `MonthlyCharges`:

| Cluster | Segment Profile | Customer Count | Avg Tenure | Avg Spend | Churn Rate | Strategic Priority |
|---|---|---|---|---|---|---|
| **0** | **Low Spending, New Customers** | 1,732 (24.6%) | 10.5 mos | $32.47 | 24.54% | Early engagement & service upsell |
| **1** | **High Spending, Long-Term (Loyal Champions)** | 1,952 (27.7%) | 58.7 mos | $93.02 | 15.63% | VIP perks & device upgrades |
| **2** | **High Spending, Short-Term (High Risk Vulnerables)** | 2,204 (31.3%) | 14.8 mos | $81.22 | **49.18%** | **CRITICAL: Emergency outreach** |
| **3** | **Low Spending, Long-Term (Steady Traditionalists)** | 1,155 (16.4%) | 54.1 mos | $34.03 | **4.76%** | Low touch; steady revenue bedrock |

---

## 🎯 7. Prescriptive Retention Playbook (Japan Market Strategy)

| Behavioral Trigger | Prescriptive Retention Action | Japanese Market Implementation |
|---|---|---|
| **Month-to-month Contract** | Annual Commitment Bonus | Offer ¥1,000/mo bill credit for switching to a 1-year contract + 3,000 loyalty points (d-point / Ponta). |
| **Tenure $\le$ 12 Months** | VIP Onboarding Program | Dedicated customer success check-in call at Day 30 and Day 90; free technical installation audit. |
| **Electronic Check Payment** | Auto-Pay Digital Incentive | One-time ¥1,500 bill rebate for registering automatic Bank Transfer or Credit Card. |
| **Fiber Optic without Tech Support** | Complimentary Support Trial | 90-day free trial of 24/7 Remote Tech Support & Online Security pack. |
| **Family / Dependents Present** | Family Plan Consolidation | *Hikari Set* bundle discount: ¥1,100 discount per family line when broadband and mobile are tied. |

---

## 💼 8. Japan Placement Interview Preparation: Questions & Answers

### Q1: What is customer churn and why is it critical in Japan?
> **Answer:** Customer churn measures the rate at which subscribers discontinue services. In Japan, with a declining population and over 100% mobile penetration, the telecom market is saturated. Customer Acquisition Cost (CAC) is 5 to 7 times higher than retention. Reducing churn protects long-term recurring revenue and Customer Lifetime Value (LTV).

### Q2: Why is Accuracy insufficient for churn prediction?
> **Answer:** Customer churn datasets have inherent class imbalance (73.5% non-churn vs. 26.5% churn). A naive model predicting "No Churn" for every customer would achieve 73.5% accuracy but zero recall, failing to save a single customer. We focus on **Recall** (77.54%) to minimize False Negatives (missed churners).

### Q3: Why choose Random Forest over Logistic Regression?
> **Answer:** Random Forest achieved a higher ROC-AUC (0.8435 vs 0.8415) and reduced False Positives by 36 cases. More importantly, it naturally captures complex non-linear interactions—such as the relationship between fiber optic internet, lack of tech support, and high bills—without manual feature cross-products.

### Q4: How do supervised classification and unsupervised clustering work together?
> **Answer:** Supervised classification predicts *who* is likely to leave and with what probability. Unsupervised K-Means clustering groups customers by *behavioral profiles* (e.g. High Spending, Short-Term). Together, they enable targeted prescriptive actions rather than blunt, expensive retention offers.

### Q5: How did you prevent data leakage during feature engineering?
> **Answer:** All engineered features (`TenureGroup`, `ServiceCount`, `AvgMonthlyCharges`, `SecurityBundle`, `StreamingBundle`) use information known strictly at prediction time. Furthermore, all statistical transformations (scaling and one-hot encoding) are encapsulated within a Scikit-learn pipeline fitted exclusively on training folds during cross-validation, guaranteeing zero leakage into test sets.

---

## 📜 9. License & Author
- Developed as a comprehensive Data Science Portfolio & Technical Interview asset.
- Code released under the MIT License.

