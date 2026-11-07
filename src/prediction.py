"""
Customer Churn Prediction and Retention Recommendation Engine
Customer Behaviour Analysis & Churn Prediction
"""

import os
import joblib
import pandas as pd
import numpy as np
from src.preprocessing import clean_data, add_engineered_features


def load_model(model_path: str):
    """Loads serialized ML pipeline model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)


def generate_retention_recommendations(customer_data: dict, churn_prob: float) -> list:
    """
    Generates actionable, prescriptive retention recommendations tailored for telecommunications,
    incorporating best practices from Japanese telco retention (point bonuses, family plans, contract incentives).
    """
    recommendations = []

    # Risk level thresholding
    if churn_prob >= 0.60:
        recommendations.append({
            "priority": "HIGH",
            "action": "Urgent Outbound Customer Success Engagement",
            "details": "Assign a dedicated Customer Care Specialist within 48 hours to assess dissatisfaction and offer an immediate retention incentive package."
        })
    elif churn_prob >= 0.35:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Proactive Loyalty Outreach & Account Review",
            "details": "Send a targeted loyalty email or in-app notification offering a free plan audit to optimize monthly costs."
        })
    else:
        recommendations.append({
            "priority": "LOW",
            "action": "Standard Relationship Nurturing",
            "details": "Maintain regular service quality checks and provide monthly loyalty points rewards."
        })

    # Contract Type optimization
    contract = customer_data.get("Contract", "")
    if contract == "Month-to-month":
        recommendations.append({
            "priority": "HIGH",
            "action": "Contract Upgrade Incentive (Annual Commitment)",
            "details": "Offer a 15% discount for switching to a 1-year or 2-year contract, bundled with 3,000 loyalty points (d-point / Ponta equivalent)."
        })

    # Tenure-based onboarding
    tenure = float(customer_data.get("tenure", 0))
    if tenure <= 12:
        recommendations.append({
            "priority": "HIGH",
            "action": "First-Year Customer Onboarding Check-in",
            "details": "Provide a VIP tech support consultation and a welcome bonus milestone after month 6 to minimize early-stage dropout."
        })

    # Payment Method friction
    payment = customer_data.get("PaymentMethod", "")
    if "Electronic check" in payment or "Mailed check" in payment:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Auto-Pay & Digital Invoicing Transition",
            "details": "Incentivize migration to recurring Credit Card or Bank Transfer Auto-Pay with a one-time bill credit ($10 / ¥1,500)."
        })

    # High Monthly Charges
    monthly_charges = float(customer_data.get("MonthlyCharges", 0))
    if monthly_charges > 75:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "High-ARPU Plan Optimization & Value Bundling",
            "details": "Recommend an integrated family or quad-play bundle discount to lower perceived bill burden while securing total contract lifetime value."
        })

    # Security & Support Add-ons
    tech_support = customer_data.get("TechSupport", "No")
    online_sec = customer_data.get("OnlineSecurity", "No")
    if tech_support == "No" or online_sec == "No":
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Complimentary Device Protection & Security Trial",
            "details": "Offer 3 months of complimentary Online Security and 24/7 Tech Support to boost product stickiness and perceived service value."
        })

    # Partner / Dependents bundle
    partner = customer_data.get("Partner", "No")
    dependents = customer_data.get("Dependents", "No")
    if partner == "Yes" or dependents == "Yes":
        recommendations.append({
            "priority": "LOW",
            "action": "Family Share Plan Promotion",
            "details": "Introduce shared family data plans and multi-line discounts to anchor household retention."
        })

    return recommendations


def predict_single_customer(model, customer_dict: dict, clustering_model=None) -> dict:
    """
    Takes customer feature dictionary, prepares features, predicts churn probability,
    assigns risk category, cluster segment, and returns retention recommendations.
    """
    df_raw = pd.DataFrame([customer_dict])
    df_clean = clean_data(df_raw)
    df_feat = add_engineered_features(df_clean)

    # Predict probability and class
    prob = float(model.predict_proba(df_feat)[0, 1])
    prediction = int(prob >= 0.5)

    if prob >= 0.65:
        risk_level = "High Churn Risk"
        risk_color = "red"
    elif prob >= 0.35:
        risk_level = "Moderate Churn Risk"
        risk_color = "orange"
    else:
        risk_level = "Low Churn Risk"
        risk_color = "green"

    # Customer segmentation
    segment = "Standard Subscriber"
    if clustering_model is not None:
        try:
            # Segment using tenure and MonthlyCharges
            seg_features = df_feat[['tenure', 'MonthlyCharges']]
            cluster_id = clustering_model.predict(seg_features)[0]
            cluster_names = {
                0: "Low Spending, New Customers (Budget Starters)",
                1: "High Spending, Long-Term (Loyal Champions)",
                2: "High Spending, Short-Term (High Risk Vulnerables)",
                3: "Low Spending, Long-Term (Steady Traditionalists)"
            }
            segment = cluster_names.get(cluster_id, f"Segment {cluster_id}")
        except Exception:
            segment = "Standard Subscriber"

    recommendations = generate_retention_recommendations(customer_dict, prob)

    return {
        "churn_prediction": "Likely to Churn" if prediction == 1 else "Likely to Stay",
        "churn_probability": prob,
        "churn_percentage": round(prob * 100, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "customer_segment": segment,
        "recommendations": recommendations
    }


def predict_batch(model, df: pd.DataFrame, clustering_model=None) -> pd.DataFrame:
    """
    Runs batch predictions on a dataframe of customers.
    """
    df_clean = clean_data(df)
    df_feat = add_engineered_features(df_clean)
    
    probs = model.predict_proba(df_feat)[:, 1]
    preds = (probs >= 0.5).astype(int)

    out_df = df.copy()
    out_df['Churn_Probability'] = np.round(probs, 4)
    out_df['Predicted_Churn'] = np.where(preds == 1, 'Yes', 'No')
    out_df['Risk_Category'] = pd.cut(
        probs,
        bins=[-0.1, 0.35, 0.65, 1.0],
        labels=['Low Risk', 'Moderate Risk', 'High Risk']
    )
    return out_df
