# Comprehensive Project Report
## Intelligent Customer Behaviour Analysis & Churn Prediction System
**Portfolio Project & Japan Data Scientist Placement Edition**

---

## 1. Executive Summary

Customer churn represents one of the most critical threats to recurring subscription revenue in modern telecommunications. In highly developed markets such as Japan, where mobile and broadband penetration exceeds 100%, the cost of customer acquisition (CAC) is between 5 to 7 times higher than retention. 

This project delivers an end-to-end **Customer Intelligence & Churn Prediction System** combining **Python, SQL, Inferential Statistics, Machine Learning (Scikit-learn), K-Means Clustering, and an interactive Streamlit Web Application**. In strict alignment with modern production best practices, the system provides three core capabilities:

1. **Predictive Analytics:** Forecasting customer churn probability with an ensemble **Random Forest** pipeline achieving **0.8435 ROC-AUC** and **77.54% Recall** on held-out test data.
2. **Customer Segmentation:** Clustering customers into four interpretable behavioral cohorts using **K-Means**, identifying an acute vulnerability group suffering a **49.2%** churn rate.
3. **Prescriptive Analytics:** Delivering automated, actionable retention playbooks customized for Japanese telecommunications (e.g., d-point/Ponta point loyalty incentives, Hikari optical fiber bundle rebates, and automated onboarding interventions).

---

## 2. Business Scenario: Japanese Telecommunications Market

### 2.1 The Market Landscape
Consider a major Japanese telecommunications operator (analogous to NTT Docomo, KDDI/au, SoftBank, or Rakuten Mobile) offering fiber optic home broadband (*Hikari*), mobile subscriptions, streaming services, and technical support.

The company observes rising churn, creating three distinct operational bottlenecks:
- **Revenue Erosion:** In our historical baseline of 7,043 customers, 1,869 churned, representing **$139,130.85/month** in direct loss (**$1.67 Million USD / ~¥250 Million JPY annually**).
- **Inability to Prioritize Outreach:** Customer care teams lack a systematic scoring mechanism to distinguish between high-risk and low-risk subscribers before cancellation occurs.
- **Unclear Root Causes:** Marketing and product teams cannot clearly quantify whether bill size, contract commitment, or lack of customer support drives customer departures.

### 2.2 Core Questions Answered
1. **Which customers are likely to leave?** Customers on month-to-month contracts, in their first 6–12 months of tenure, using electronic checks, and lacking technical support.
2. **Why are customers leaving?** Contract friction, early onboarding dissonance, and bill shock on fiber optic services without adequate technical support.
3. **Which services have the highest churn?** Fiber optic internet subscribers without Tech Support churn at **49.37%** (compared to only 22.63% when Tech Support is present).
4. **Which customer groups are most vulnerable?** *High Spending, Short-Term Customers* (Cluster 2) exhibit a **49.18%** churn rate.
5. **What actions can retain customers?** 1-year contract extension bonuses with loyalty points, onboarding care for the first 90 days, automated payment discounts, and complimentary tech support bundles.

---

## 3. Dataset Profiling & Preprocessing

- **Dataset:** IBM Telco Customer Churn Dataset (7,043 rows, 21 columns).
- **Target Variable:** `Churn` (Binary: Yes = 1, No = 0).
- **Class Balance:** Retained: 5,174 (73.46%), Churned: 1,869 (26.54%).

### 3.1 Data Cleansing
- **TotalCharges Blank String Handling:** 11 records contained blank string whitespace (`" "`) in `TotalCharges`. Inspection revealed that all 11 had `tenure = 0` (brand-new customers). These were logically imputed with `0.0`.
- **Identifier Removal:** `customerID` was removed from features to prevent memorization.
- **Leakage-Free Architecture:** Scikit-learn's `ColumnTransformer` was fitted strictly on training data during cross-validation, using `StandardScaler` for numeric columns and `OneHotEncoder(handle_unknown='ignore')` for categorical columns.

---

## 4. Exploratory Data Analysis & Inferential Statistics

### 4.1 Key Exploratory Findings
1. **Contract Dynamics:** Month-to-month customers exhibit a **42.71%** churn rate, whereas 1-year contracts drop to **11.27%**, and 2-year contracts drop to **2.83%**.
2. **Tenure Hazard Curve:** 0–6 months has a **52.94%** churn rate. Churn risk decreases exponentially as customer tenure lengthens.
3. **Payment Friction:** Electronic check users churn at **45.29%**, compared to ~**15%** for automated credit card and bank transfer subscribers.
4. **Service Stickiness:** Customers subscribing to 8 or 9 services churn at only **12.4%** and **5.3%**, showing that service ecosystem integration insulates customers against cancellation.

### 4.2 Formal Hypothesis Testing (SciPy)
To establish mathematical rigor, statistical hypothesis tests were conducted:

| Test Type | Variables Tested | Test Statistic | p-value | Conclusion |
|---|---|---|---|---|
| **Chi-Square Test** | Contract vs Churn | $\chi^2 = 1184.6$ | $p < 10^{-200}$ | Reject $H_0$; highly significant association |
| **Chi-Square Test** | Payment Method vs Churn | $\chi^2 = 648.4$ | $p < 10^{-140}$ | Reject $H_0$; electronic check drives churn |
| **Chi-Square Test** | Internet Service vs Churn | $\chi^2 = 732.3$ | $p < 10^{-150}$ | Reject $H_0$; fiber optic customers churn more |
| **Two-Sample T-Test** | Monthly Charges (Stay vs Churn) | $t = 18.41$ | $p < 10^{-70}$ | Churners pay significantly higher monthly charges |
| **Two-Sample T-Test** | Tenure (Stay vs Churn) | $t = -34.82$ | $p < 10^{-200}$ | Churners have significantly lower tenure |

---

## 5. Feature Engineering

Domain-specific features were engineered without data leakage:
1. `TenureGroup`: Categorized into `'New Customer'` (0–12m), `'Regular Customer'` (13–36m), and `'Loyal Customer'` (>36m).
2. `ServiceCount`: Total count of active subscribed services across mobile lines, internet, security, backup, device protection, tech support, and streaming.
3. `AvgMonthlyCharges`: Ratio of `TotalCharges / max(tenure, 1)` capturing expenditure acceleration.
4. `SecurityBundle`: Boolean indicator for concurrent `OnlineSecurity` and `TechSupport` adoption.
5. `StreamingBundle`: Boolean indicator for concurrent `StreamingTV` and `StreamingMovies` adoption.

---

## 6. SQL Analytics Engine

The accompanying SQLite database (`telco_churn.db`) and SQL script (`sql/customer_analysis.sql`) execute eight analytical queries:
1. Overall Churn and MRR loss.
2. Revenue at risk segmented by contract length.
3. Tenure cohort breakdown across 5 lifecycle stages.
4. Fiber optic vs. DSL tech support interaction.
5. Payment method churn rates.
6. High-value customer identification (ARPU $\ge \$80$/mo).
7. Service bundling penetration.
8. Demographic analysis (Senior citizens and family bundles).

---

## 7. Machine Learning Model Development & Benchmark

Four classification algorithms were trained using identical 80/20 stratified splits and 5-fold cross-validation. In compliance with project constraints, neither XGBoost nor SHAP was used.

### Model Performance on 20% Held-Out Test Set (1,409 customers):

| Algorithm | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score | ROC-AUC | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** | **76.30%** | **53.70%** | **77.54%** | **0.6346** | **0.8435** | 290 | 84 | 250 | 785 |
| **Decision Tree** | 74.24% | 50.94% | 79.41% | 0.6207 | 0.8344 | 297 | 77 | 286 | 749 |
| **Logistic Regression** | 73.95% | 50.60% | 78.88% | 0.6165 | 0.8415 | 295 | 79 | 288 | 747 |
| **K-Nearest Neighbors** | 75.80% | 54.65% | 51.87% | 0.5322 | 0.7871 | 194 | 180 | 161 | 874 |

### Why Random Forest is the Selected Production Model:
1. **Best Overall Discrimination:** Highest ROC-AUC score (**0.8435**).
2. **Superior Precision-Recall Trade-off:** Achieves **77.54% Recall** while committing **36 fewer False Positives** than Logistic Regression and single Decision Trees.
3. **Capture of Nonlinear Feature Interactions:** Effortlessly captures complex interactions (e.g., Fiber Optic + Month-to-Month + No Tech Support) without requiring explicit interaction term engineering.

### Top Predictive Features (Random Forest Feature Importance):
1. `Contract_Month-to-month` (12.57%)
2. `tenure` (9.64%)
3. `TotalCharges` (6.92%)
4. `OnlineSecurity_No` (6.31%)
5. `Contract_Two year` (5.72%)
6. `InternetService_Fiber optic` (5.33%)
7. `TechSupport_No` (4.66%)
8. `AvgMonthlyCharges` (4.52%)
9. `MonthlyCharges` (4.26%)
10. `PaymentMethod_Electronic check` (3.71%)

---

## 8. Customer Segmentation (K-Means Clustering)

Using standardized `tenure` and `MonthlyCharges`, customers were grouped into four behavioral segments:

| Cluster | Segment Name | Customer Count | Avg Tenure | Avg Monthly Spend | Churn Rate | Strategic Retention Objective |
|---|---|---|---|---|---|---|
| **Cluster 0** | **Low Spending, New Customers** | 1,732 (24.6%) | 10.5 mos | $32.47 | 24.54% | Early engagement & service upsell |
| **Cluster 1** | **High Spending, Long-Term (Loyal Champions)** | 1,952 (27.7%) | 58.7 mos | $93.02 | 15.63% | VIP loyalty perks & device upgrade renewal |
| **Cluster 2** | **High Spending, Short-Term (High Risk Vulnerables)** | 2,204 (31.3%) | 14.8 mos | $81.22 | **49.18%** | **CRITICAL: Emergency outreach & onboarding care** |
| **Cluster 3** | **Low Spending, Long-Term (Steady Traditionalists)** | 1,155 (16.4%) | 54.1 mos | $34.03 | **4.76%** | Low touch; steady revenue bedrock |

---

## 9. Prescriptive Retention Playbook (Japan Market Strategy)

| Customer Characteristic | Prescriptive Action Plan | Japanese Telco Implementation |
|---|---|---|
| **Month-to-month Contract** | Annual Commitment Bonus | Offer ¥1,000/mo bill credit for switching to a 1-year contract, bundled with 3,000 loyalty points (d-point / Ponta). |
| **Tenure $\le$ 12 Months** | VIP Onboarding Program | Proactive customer success check-in call at Day 30 and Day 90; free technical installation audit. |
| **Electronic Check Payment** | Auto-Pay Digital Incentive | One-time ¥1,500 bill rebate for registering automatic Bank Transfer or Credit Card. |
| **Fiber Optic without Tech Support** | Complimentary Support Trial | 90-day free trial of 24/7 Remote Tech Support & Online Security pack. |
| **Family / Dependents Present** | Family Plan Consolidation | *Hikari Set* bundle discount: ¥1,100 discount per family line when broadband and mobile are tied. |

---

## 10. Deployment: Streamlit Web Application

The interactive web application (`dashboard/app.py`) provides four functional pages:
- **Page 1: Business Overview:** High-level KPIs, monthly/annual revenue loss calculations, and an interactive ROI simulator showing revenue saved per percentage point of churn reduction.
- **Page 2: Customer Analytics:** Interactive charts for contract churn, tenure hazard curves, payment friction, and K-Means segmentation scatter plots.
- **Page 3: Churn Prediction & Retention:** Interactive single-customer profiler with instant probability scoring, segment assignment, and personalized Japanese market retention playbooks. Includes batch CSV upload and download capability.
- **Page 4: Model Performance:** Side-by-side benchmark comparison of all four algorithms, interactive confusion matrices, multi-model ROC curve plots, and feature importance rankings.

---

## 11. Japan Placement Interview Preparation: Questions & Answers

### Q1: Why is customer churn analysis so critical for Japanese companies?
> **Answer:** In Japan, population demographics and market saturation mean telecommunications and subscription markets are zero-sum games. Acquiring a new customer costs 5 to 7 times more than retaining an existing one. Furthermore, customer loyalty in Japan is closely linked to service quality (*omotenashi*). Retaining customers protects long-term Customer Lifetime Value (LTV) and brand reputation.

### Q2: Why did you not use Accuracy as your primary evaluation metric?
> **Answer:** The dataset has a 73.5% / 26.5% class imbalance. A trivial majority-class classifier that predicts 'No Churn' for everyone would achieve 73.5% accuracy but zero recall, failing completely at the business task. We prioritized **Recall on Churn** (77.54%) because a False Negative means an unidentified departing customer, leading to direct recurring revenue loss.

### Q3: Why did you choose Random Forest over Logistic Regression and single Decision Trees?
> **Answer:** Random Forest achieved the highest ROC-AUC (0.8435) and effectively reduced variance compared to individual decision trees. It naturally captures nonlinear interactions—such as high monthly charges combined with fiber optic internet and absence of tech support—without needing manual interaction features. Furthermore, it reduced False Positives by 36 cases compared to Logistic Regression, lowering unnecessary retention marketing costs.

### Q4: How does your K-Means clustering complement your supervised classification model?
> **Answer:** Classification answers *who* is likely to leave and with what probability. Clustering answers *what type of customer* they are. Our K-Means model identified that **Cluster 2 (High Spending, Short-Term)** accounts for nearly half of all churned revenue. This allows the business to deliver targeted, segment-specific retention campaigns rather than generic discounts.

### Q5: How did you ensure your feature engineering avoided data leakage?
> **Answer:** All engineered features (`TenureGroup`, `ServiceCount`, `AvgMonthlyCharges`, `SecurityBundle`, `StreamingBundle`) use information known strictly at prediction time. Furthermore, all statistical transformations (scaling and one-hot encoding) are encapsulated within a Scikit-learn pipeline fitted exclusively on training folds during cross-validation, guaranteeing zero leakage into test sets.

