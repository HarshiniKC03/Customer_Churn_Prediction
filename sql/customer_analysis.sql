-- ============================================================================
-- SQL Customer Analysis Queries: Customer Behaviour & Churn Prediction
-- Japanese Telco Business Case Analysis
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. OVERALL CHURN RATE AND MONTHLY REVENUE LOSS
-- Business Goal: Measure overall churn percentage and monthly recurring revenue lost.
-- ----------------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS churned_customers,
    SUM(CASE WHEN Churn = 0 THEN 1 ELSE 0 END) AS retained_customers,
    ROUND(AVG(Churn) * 100, 2) AS churn_rate_percentage,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_revenue,
    ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END), 2) AS lost_monthly_revenue,
    ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END) * 12, 2) AS estimated_annual_revenue_loss
FROM customers;


-- ----------------------------------------------------------------------------
-- 2. CHURN RATE & REVENUE LOSS BY CONTRACT TYPE
-- Business Goal: Test hypothesis whether Month-to-month contracts drive majority of churn.
-- ----------------------------------------------------------------------------
SELECT 
    Contract,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS churned_count,
    ROUND(AVG(Churn) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge,
    ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_at_risk
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- 3. TENURE COHORT ANALYSIS
-- Business Goal: Identify early customer drop-off period to guide onboarding retention.
-- ----------------------------------------------------------------------------
SELECT 
    CASE 
        WHEN tenure <= 6 THEN '01. 0 - 6 Months (Early Onboarding)'
        WHEN tenure <= 12 THEN '02. 7 - 12 Months (First Year)'
        WHEN tenure <= 24 THEN '03. 13 - 24 Months (Second Year)'
        WHEN tenure <= 48 THEN '04. 25 - 48 Months (Established)'
        ELSE '05. 49+ Months (Long-Term Loyal)'
    END AS tenure_cohort,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS churned_count,
    ROUND(AVG(Churn) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_bill
FROM customers
GROUP BY tenure_cohort
ORDER BY tenure_cohort ASC;


-- ----------------------------------------------------------------------------
-- 4. INTERNET SERVICE TYPE & TECH SUPPORT IMPACT
-- Business Goal: Evaluate whether Fiber optic churn is tied to lack of Tech Support.
-- ----------------------------------------------------------------------------
SELECT 
    InternetService,
    TechSupport,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS churned_count,
    ROUND(AVG(Churn) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_bill
FROM customers
GROUP BY InternetService, TechSupport
ORDER BY InternetService, churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- 5. PAYMENT METHOD & FRICTION ANALYSIS
-- Business Goal: Analyze if non-automated payment methods (e.g. Electronic check) have higher friction.
-- ----------------------------------------------------------------------------
SELECT 
    PaymentMethod,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS churned_count,
    ROUND(AVG(Churn) * 100, 2) AS churn_rate_pct,
    ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END), 2) AS monthly_lost_revenue
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- 6. HIGH-VALUE CUSTOMER CHURN RISK (ARPU >= $80)
-- Business Goal: Identify VIP customers at high risk to prioritize high-touch outreach.
-- ----------------------------------------------------------------------------
SELECT 
    CustomerID,
    Contract,
    tenure,
    InternetService,
    MonthlyCharges,
    TotalCharges,
    PaymentMethod
FROM customers
WHERE MonthlyCharges >= 80 
  AND Churn = 1
  AND Contract = 'Month-to-month'
ORDER BY MonthlyCharges DESC
LIMIT 15;


-- ----------------------------------------------------------------------------
-- 7. MULTI-SERVICE BUNDLING EFFECT (STICKINESS ANALYSIS)
-- Business Goal: Check if customers who subscribe to multiple services are more loyal.
-- ----------------------------------------------------------------------------
SELECT 
    ServiceCount,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS churned_count,
    ROUND(AVG(Churn) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_bill
FROM customers
GROUP BY ServiceCount
ORDER BY ServiceCount ASC;


-- ----------------------------------------------------------------------------
-- 8. SENIOR CITIZEN & DEPENDENTS RETENTION OPPORTUNITY
-- Business Goal: Evaluate family plan opportunities in Japanese market context.
-- ----------------------------------------------------------------------------
SELECT 
    SeniorCitizen,
    Partner,
    Dependents,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS churned_count,
    ROUND(AVG(Churn) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY SeniorCitizen, Partner, Dependents
ORDER BY churn_rate_pct DESC;

