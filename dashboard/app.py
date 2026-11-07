"""
Streamlit Web Application
Customer Behaviour Analysis & Churn Prediction System
Intelligent Japanese Telco Portfolio Project
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Add parent directory to sys.path so we can import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.preprocessing import clean_data, add_engineered_features, load_raw_data
from src.prediction import predict_single_customer, predict_batch, load_model

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Intelligence & Churn Prediction System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-med {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data & Model Caching
# -----------------------------------------------------------------------------
@st.cache_data
def get_dataset():
    data_path = os.path.join(parent_dir, 'dataset', 'customer_churn.csv')
    df_raw = load_raw_data(data_path)
    df_clean = clean_data(df_raw)
    df_feat = add_engineered_features(df_clean)
    return df_raw, df_feat


@st.cache_resource
def get_models():
    model_path = os.path.join(parent_dir, 'models', 'churn_model.pkl')
    cluster_path = os.path.join(parent_dir, 'models', 'clustering_model.pkl')
    churn_model = load_model(model_path)
    clustering_model = load_model(cluster_path)
    return churn_model, clustering_model


@st.cache_data
def get_evaluation_metadata():
    eval_path = os.path.join(parent_dir, 'models', 'evaluation_results.json')
    comp_path = os.path.join(parent_dir, 'models', 'model_comparison.csv')
    feat_path = os.path.join(parent_dir, 'models', 'feature_importance.csv')
    seg_path = os.path.join(parent_dir, 'models', 'segment_summary.csv')
    
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)
    comp_df = pd.read_csv(comp_path)
    feat_df = pd.read_csv(feat_path)
    seg_df = pd.read_csv(seg_path)
    return eval_data, comp_df, feat_df, seg_df


# Load resources
try:
    df_raw, df_feat = get_dataset()
    churn_model, clustering_model = get_models()
    eval_data, comp_df, feat_df, seg_df = get_evaluation_metadata()
except Exception as e:
    st.error(f"Error loading assets: {e}. Please ensure train_models.py has been executed.")
    st.stop()


# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/telecommunication-antenna.png", width=70)
st.sidebar.title("Customer Intelligence")
st.sidebar.markdown("**Portfolio & Placement Edition**")

page = st.sidebar.radio(
    "Navigate to Module:",
    [
        "🏢 1. Business Overview",
        "📊 2. Customer Analytics",
        "🎯 3. Churn Prediction & Retention",
        "📈 4. Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Domain Context:**
- Telecom & Subscription Services
- Target Market: Japan Telco Industry
- Key Technologies: Scikit-learn, Pandas, SciPy, Streamlit
- Note: Pure Scikit-learn ML (No XGBoost / No SHAP)
""")


# =============================================================================
# PAGE 1: BUSINESS OVERVIEW
# =============================================================================
if page == "🏢 1. Business Overview":
    st.markdown('<div class="main-title">Intelligent Customer Behaviour & Churn Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Telecommunications Customer Retention System | Japan Market Business Case</div>', unsafe_allow_html=True)

    # Core Metrics Calculation
    total_customers = len(df_raw)
    churned_customers = (df_feat['Churn'] == 1).sum()
    active_customers = total_customers - churned_customers
    churn_rate = (churned_customers / total_customers) * 100
    avg_monthly_charges = df_feat['MonthlyCharges'].mean()
    monthly_revenue_lost = df_feat[df_feat['Churn'] == 1]['MonthlyCharges'].sum()
    annual_revenue_lost = monthly_revenue_lost * 12
    # In JPY at ~150 JPY/USD
    annual_lost_jpy = annual_revenue_lost * 150

    # KPI Cards Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">{total_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Customers</div>
            <div class="metric-value" style="color: #16A34A;">{active_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Churned Customers</div>
            <div class="metric-value" style="color: #DC2626;">{churned_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall Churn Rate</div>
            <div class="metric-value" style="color: #DC2626;">{churn_rate:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Monthly Bill</div>
            <div class="metric-value">${avg_monthly_charges:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue Impact & Retention Opportunity
    row2_col1, row2_col2 = st.columns([3, 2])
    with row2_col1:
        st.subheader("Financial Impact & Revenue at Risk")
        st.write(
            "Customer churn directly undermines recurring subscription revenue. "
            "In telecommunications, acquiring a new customer costs 5 to 7 times more than retaining an existing customer."
        )
        fin_col1, fin_col2 = st.columns(2)
        with fin_col1:
            st.metric("Monthly Recurring Revenue Lost", f"${monthly_revenue_lost:,.2f}")
        with fin_col2:
            st.metric("Estimated Annual Loss (USD / JPY)", f"${annual_revenue_lost:,.2f}", f"¥{annual_lost_jpy/1e6:.1f}M JPY")

        st.markdown("""
        > **Strategic Takeaway for Japan Telco Operations:**
        > Reducing churn by merely **5 percentage points** would preserve **$83,478 USD (approx. ¥12.5M JPY)** in annual recurring subscription revenue, while protecting customer lifetime value (LTV).
        """)

    with row2_col2:
        st.subheader("Retention Opportunity Simulator")
        reduction_target = st.slider("Target Churn Reduction (%)", min_value=1, max_value=25, value=5, step=1)
        saved_customers = int(churned_customers * (reduction_target / 100))
        saved_annual_rev = saved_customers * avg_monthly_charges * 12
        saved_jpy = saved_annual_rev * 150

        st.success(f"""
        **Projected Business ROI:**
        - **Retained Customers:** {saved_customers:,} subscribers
        - **Annual Revenue Saved:** ${saved_annual_rev:,.2f}
        - **Japanese Yen Impact:** ¥{saved_jpy:,.0f} JPY
        """)

    st.markdown("---")
    st.subheader("Five Core Business Questions Answered")
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        st.markdown("""
        1. **Which customers are likely to leave?**
           - Customers on month-to-month contracts, in their first 6-12 months of tenure, paying through electronic check, and using fiber optic internet without technical support.
        2. **Why are customers leaving?**
           - Lack of contract commitment, onboarding friction, higher billing without technical assistance, and payment method friction.
        3. **Which services have highest customer churn?**
           - Fiber optic internet subscribers without Tech Support experience a staggering **49.4%** churn rate.
        """)
    with q_col2:
        st.markdown("""
        4. **Which customer groups are most vulnerable?**
           - *High Spending, Short-Term Customers* have an alarming churn rate of **49.2%**, representing the greatest immediate financial exposure.
        5. **What actions should the company take to retain them?**
           - Annual contract upgrade incentives with point credits (d-point/Ponta style), proactive 90-day onboarding, automatic bank transfer discounts, and complimentary tech support bundles.
        """)


# =============================================================================
# PAGE 2: CUSTOMER ANALYTICS & SEGMENTATION
# =============================================================================
elif page == "📊 2. Customer Analytics":
    st.markdown('<div class="main-title">Exploratory Data Analysis & Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Uncovering behavioural patterns, contract dynamics, and K-Means customer clusters</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Contract & Tenure Dynamics",
        "💳 Billing & Payment Friction",
        "🌐 Services & Bundle Stickiness",
        "👥 K-Means Customer Segments"
    ])

    with tab1:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Churn Rate by Contract Type")
            contract_churn = df_feat.groupby('Contract')['Churn'].agg(
                Total='count',
                Churn_Rate=lambda x: (x.mean() * 100)
            ).reset_index()

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(contract_churn['Contract'], contract_churn['Churn_Rate'], color=['#DC2626', '#3B82F6', '#10B981'])
            ax.set_ylabel("Churn Rate (%)")
            ax.set_ylim(0, 50)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            st.info("💡 **Insight:** Month-to-month contracts have a **42.7%** churn rate, compared to just **2.8%** for 2-year contracts. Long contracts serve as vital retention anchors.")

        with col_c2:
            st.markdown("#### Churn Rate by Tenure Group")
            tenure_churn = df_feat.groupby('TenureGroup')['Churn'].agg(
                Total='count',
                Churn_Rate=lambda x: (x.mean() * 100)
            ).reindex(['New Customer', 'Regular Customer', 'Loyal Customer']).reset_index()

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(tenure_churn['TenureGroup'], tenure_churn['Churn_Rate'], color=['#EF4444', '#F59E0B', '#10B981'])
            ax.set_ylabel("Churn Rate (%)")
            ax.set_ylim(0, 55)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            st.info("💡 **Insight:** New customers (<=12 months) churn at **47.7%**. Once a customer reaches 36+ months, churn drops to **14.0%**.")

    with tab2:
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("#### Payment Method vs. Churn Rate")
            pm_churn = df_feat.groupby('PaymentMethod')['Churn'].agg(
                Churn_Rate=lambda x: (x.mean() * 100)
            ).sort_values(by='Churn_Rate', ascending=False).reset_index()

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(pm_churn['PaymentMethod'], pm_churn['Churn_Rate'], color='#3B82F6')
            ax.set_xlabel("Churn Rate (%)")
            for bar in bars:
                width = bar.get_width()
                ax.annotate(f'{width:.1f}%',
                            xy=(width, bar.get_y() + bar.get_height() / 2),
                            xytext=(5, 0), textcoords="offset points",
                            ha='left', va='center', fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            st.info("💡 **Insight:** Electronic Check payments have a **45.3%** churn rate, while automated Credit Card and Bank Transfer churn is ~**15%**.")

        with col_b2:
            st.markdown("#### Monthly Charges Distribution (Stay vs. Churn)")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df_feat[df_feat['Churn'] == 0]['MonthlyCharges'], label='Stayed', color='#10B981', kde=True, ax=ax, alpha=0.4)
            sns.histplot(df_feat[df_feat['Churn'] == 1]['MonthlyCharges'], label='Churned', color='#EF4444', kde=True, ax=ax, alpha=0.4)
            ax.set_xlabel("Monthly Charges ($)")
            ax.set_ylabel("Customer Count")
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            st.info("💡 **Insight:** Churn spikes between $70 and $105/month, where customers often experience 'bill shock' if they don't perceive sufficient value.")

    with tab3:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### Internet Service & Tech Support Interaction")
            int_tech = df_feat.groupby(['InternetService', 'TechSupport'])['Churn'].agg(
                Churn_Rate=lambda x: (x.mean() * 100)
            ).reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=int_tech, x='InternetService', y='Churn_Rate', hue='TechSupport', palette='Set2', ax=ax)
            ax.set_ylabel("Churn Rate (%)")
            ax.set_title("Tech Support Protects Fiber Customers")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            st.info("💡 **Insight:** Fiber optic customers without Tech Support churn at **49.4%**, but adding Tech Support cuts that rate down to **22.6%**!")

        with col_s2:
            st.markdown("#### Multi-Service Bundling & Stickiness")
            bundle_churn = df_feat.groupby('ServiceCount')['Churn'].agg(
                Churn_Rate=lambda x: (x.mean() * 100)
            ).reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(bundle_churn['ServiceCount'], bundle_churn['Churn_Rate'], marker='o', linewidth=2.5, color='#8B5CF6')
            ax.set_xlabel("Number of Subscribed Services (1 to 9)")
            ax.set_ylabel("Churn Rate (%)")
            ax.set_title("Service Adoption Curve")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            st.info("💡 **Insight:** Customers with 7+ services have churn rates below **15%**, demonstrating that multi-service stickiness prevents churn despite higher bills.")

    with tab4:
        st.markdown("#### K-Means Customer Segmentation (4 Behavioural Clusters)")
        st.write("Customers are grouped into 4 distinct segments based on their tenure and monthly spend:")
        
        # Display Segments Table
        st.dataframe(
            seg_df.style.format({
                'Customer_Count': '{:,}',
                'Avg_Tenure': '{:.1f} months',
                'Avg_Monthly_Charges': '${:.2f}',
                'Churn_Rate': '{:.1%}'
            }),
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        # Add Cluster labels to feat_df
        cluster_labels = clustering_model.predict(df_feat[['tenure', 'MonthlyCharges']])
        palette = ['#3B82F6', '#10B981', '#EF4444', '#F59E0B']
        scatter = sns.scatterplot(
            x=df_feat['tenure'],
            y=df_feat['MonthlyCharges'],
            hue=cluster_labels,
            palette=palette,
            alpha=0.6,
            ax=ax
        )
        ax.set_xlabel("Tenure (Months)")
        ax.set_ylabel("Monthly Charges ($)")
        ax.set_title("Customer Segmentation Map (Tenure vs. Monthly Charges)")
        ax.legend(
            title="Customer Cluster",
            labels=[
                "0: Budget Starters (Low Spend, Short Tenure)",
                "1: Loyal Champions (High Spend, Long Tenure)",
                "2: High Risk Vulnerables (High Spend, Short Tenure)",
                "3: Steady Traditionalists (Low Spend, Long Tenure)"
            ]
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)


# =============================================================================
# PAGE 3: CHURN PREDICTION & RETENTION ACTION ENGINE
# =============================================================================
elif page == "🎯 3. Churn Prediction & Retention":
    st.markdown('<div class="main-title">Interactive Churn Prediction & Retention Advisory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Assess real-time customer churn probability and generate prescriptive Japanese telco retention playbooks</div>', unsafe_allow_html=True)

    mode = st.radio("Choose Prediction Mode:", ["Single Customer Profiling", "Batch CSV Scoring"], horizontal=True)

    if mode == "Single Customer Profiling":
        # Sample profiles for quick testing
        st.markdown("##### Quick-Load Preset Profiles:")
        col_p1, col_p2, col_p3 = st.columns(3)
        sample_choice = None
        if col_p1.button("🚨 High-Risk New Fiber User"):
            sample_choice = "high_risk"
        if col_p2.button("🛡️ Loyal Multi-Year VIP"):
            sample_choice = "loyal_vip"
        if col_p3.button("⚖️ Moderate-Risk Budget User"):
            sample_choice = "moderate"

        # Defaults based on sample choice
        if sample_choice == "high_risk":
            def_gender, def_sn, def_pt, def_dp = "Female", "0", "No", "No"
            def_tenure, def_ps, def_ml = 3, "Yes", "No"
            def_is, def_os, def_ob, def_dp_tech, def_ts = "Fiber optic", "No", "No", "No", "No"
            def_tv, def_mv = "Yes", "No"
            def_contract, def_pb, def_pm = "Month-to-month", "Yes", "Electronic check"
            def_mc, def_tc = 85.50, 256.50
        elif sample_choice == "loyal_vip":
            def_gender, def_sn, def_pt, def_dp = "Male", "0", "Yes", "Yes"
            def_tenure, def_ps, def_ml = 60, "Yes", "Yes"
            def_is, def_os, def_ob, def_dp_tech, def_ts = "Fiber optic", "Yes", "Yes", "Yes", "Yes"
            def_tv, def_mv = "Yes", "Yes"
            def_contract, def_pb, def_pm = "Two year", "No", "Credit card (automatic)"
            def_mc, def_tc = 105.00, 6300.00
        else:
            def_gender, def_sn, def_pt, def_dp = "Male", "0", "No", "No"
            def_tenure, def_ps, def_ml = 12, "Yes", "No"
            def_is, def_os, def_ob, def_dp_tech, def_ts = "DSL", "No", "Yes", "No", "No"
            def_tv, def_mv = "No", "No"
            def_contract, def_pb, def_pm = "Month-to-month", "Yes", "Mailed check"
            def_mc, def_tc = 50.00, 600.00

        with st.form("customer_input_form"):
            st.markdown("#### Customer Attributes")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Demographics**")
                gender = st.selectbox("Gender", ["Female", "Male"], index=0 if def_gender=="Female" else 1)
                senior = st.selectbox("Senior Citizen", ["0", "1"], index=0 if def_sn=="0" else 1, help="1 = Yes (65+ years old), 0 = No")
                partner = st.selectbox("Partner", ["No", "Yes"], index=0 if def_pt=="No" else 1)
                dependents = st.selectbox("Dependents", ["No", "Yes"], index=0 if def_dp=="No" else 1)
                tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=def_tenure)

            with c2:
                st.markdown("**Service Subscriptions**")
                phone = st.selectbox("Phone Service", ["Yes", "No"], index=0 if def_ps=="Yes" else 1)
                multilines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=0 if def_ml=="No" else (1 if def_ml=="Yes" else 2))
                internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=0 if def_is=="Fiber optic" else (1 if def_is=="DSL" else 2))
                online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=0 if def_os=="No" else 1)
                online_bk = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=0 if def_ob=="No" else 1)
                dev_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=0 if def_dp_tech=="No" else 1)
                tech_supp = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=0 if def_ts=="No" else 1)
                stream_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=0 if def_tv=="No" else 1)
                stream_mov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=0 if def_mv=="No" else 1)

            with c3:
                st.markdown("**Contract & Billing**")
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(def_contract))
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if def_pb=="Yes" else 1)
                payment = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(def_pm)
                )
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=float(def_mc), step=1.0)
                calc_total = tenure * monthly_charges
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=float(calc_total), step=10.0)

            submitted = st.form_submit_button("⚡ Predict Churn Risk & Generate Action Plan", use_container_width=True)

        if submitted:
            cust_dict = {
                'gender': gender,
                'SeniorCitizen': senior,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone,
                'MultipleLines': multilines,
                'InternetService': internet,
                'OnlineSecurity': online_sec,
                'OnlineBackup': online_bk,
                'DeviceProtection': dev_prot,
                'TechSupport': tech_supp,
                'StreamingTV': stream_tv,
                'StreamingMovies': stream_mov,
                'Contract': contract,
                'PaperlessBilling': paperless,
                'PaymentMethod': payment,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }

            res = predict_single_customer(churn_model, cust_dict, clustering_model)

            st.markdown("---")
            st.markdown("### Prediction Result & Risk Assessment")

            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                badge_class = "badge-high" if res['risk_level'] == "High Churn Risk" else ("badge-med" if res['risk_level'] == "Moderate Churn Risk" else "badge-low")
                st.markdown(f"**Predicted Outcome:** <span class='{badge_class}'>{res['churn_prediction']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Risk Category:** {res['risk_level']}")
            with res_col2:
                st.metric("Churn Probability", f"{res['churn_percentage']}%")
                st.progress(float(res['churn_probability']))
            with res_col3:
                st.markdown(f"**Customer Segment:**")
                st.info(f"{res['customer_segment']}")

            st.markdown("#### 🎯 Prescriptive Retention Playbook (Japan Market Strategy)")
            for rec in res['recommendations']:
                p_badge = "badge-high" if rec['priority'] == "HIGH" else ("badge-med" if rec['priority'] == "MEDIUM" else "badge-low")
                st.markdown(f"""
                - <span class="{p_badge}">{rec['priority']} PRIORITY</span> **{rec['action']}**  
                  *{rec['details']}*
                """, unsafe_allow_html=True)

    else:
        st.markdown("#### Batch CSV Churn Prediction")
        st.write("Upload a customer CSV file to score all customers simultaneously.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} rows. Processing predictions...")
            scored_df = predict_batch(churn_model, batch_df, clustering_model)
            
            st.success("Scoring complete! Preview of results:")
            st.dataframe(scored_df[['customerID' if 'customerID' in scored_df.columns else scored_df.columns[0], 'tenure', 'Contract', 'MonthlyCharges', 'Churn_Probability', 'Predicted_Churn', 'Risk_Category']].head(15))
            
            csv_data = scored_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored CSV",
                data=csv_data,
                file_name="scored_customers_churn.csv",
                mime="text/csv"
            )


# =============================================================================
# PAGE 4: MODEL PERFORMANCE & EVALUATION
# =============================================================================
elif page == "📈 4. Model Performance":
    st.markdown('<div class="main-title">Machine Learning Model Evaluation & Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Rigorous benchmarking of 4 algorithms with emphasis on Churn Recall and ROC-AUC</div>', unsafe_allow_html=True)

    st.markdown("#### Candidate Model Benchmark Comparison (Test Set - 20% Holdout)")
    st.dataframe(
        comp_df.style.highlight_max(subset=['Accuracy', 'Recall (Churn)', 'ROC-AUC', 'F1-Score'], color='#D1FAE5')
                    .format({
                        'Accuracy': '{:.2%}',
                        'Precision (Churn)': '{:.2%}',
                        'Recall (Churn)': '{:.2%}',
                        'F1-Score': '{:.4f}',
                        'ROC-AUC': '{:.4f}'
                    }),
        use_container_width=True
    )

    st.markdown("""
    > **Model Selection Rationale for Japan Placement Interviews:**
    > - While **Accuracy** for a trivial baseline model might look high, it completely fails to catch churners in imbalanced datasets.
    > - We prioritize **Recall on Churn** because every undetected churner causes direct revenue loss.
    > - **Random Forest** achieved the best overall balance (**ROC-AUC 0.8435**, **Recall 77.5%**, and **Accuracy 76.3%**), while significantly reducing False Positives compared to single Decision Trees.
    """)

    st.markdown("---")
    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("#### Confusion Matrix (Selected Model)")
        selected_model_name = st.selectbox("Choose Model:", list(eval_data.keys()), index=2)
        cm_data = eval_data[selected_model_name]['confusion_matrix']
        cm_matrix = np.array([
            [cm_data['tn'], cm_data['fp']],
            [cm_data['fn'], cm_data['tp']]
        ])

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            xticklabels=['Predicted: Stay', 'Predicted: Churn'],
            yticklabels=['Actual: Stay', 'Actual: Churn'],
            ax=ax
        )
        ax.set_ylabel("Actual Class")
        ax.set_xlabel("Predicted Class")
        ax.set_title(f"Confusion Matrix: {selected_model_name}")
        st.pyplot(fig)

        st.write(f"""
        - **True Positives (Caught Churners):** {cm_data['tp']}
        - **False Negatives (Missed Churners):** {cm_data['fn']}
        - **False Positives (False Alarms):** {cm_data['fp']}
        - **True Negatives (Correctly Retained):** {cm_data['tn']}
        """)

    with m_col2:
        st.markdown("#### ROC Curves Comparison")
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, metrics in eval_data.items():
            if metrics['fpr'] is not None and metrics['tpr'] is not None:
                ax.plot(metrics['fpr'], metrics['tpr'], label=f"{name} (AUC = {metrics['roc_auc']:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC = 0.50)')
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("Receiver Operating Characteristic (ROC)")
        ax.legend(loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### Top Features Driving Customer Churn (Random Forest Importance)")
    fig, ax = plt.subplots(figsize=(10, 5))
    top_feats = feat_df.head(10).iloc[::-1]
    # Clean feature names for presentation
    clean_labels = [
        f.replace('cat__', '').replace('num__', '').replace('_', ' ')
        for f in top_feats['Feature']
    ]
    ax.barh(clean_labels, top_feats['Importance'], color='#3B82F6')
    ax.set_xlabel("Feature Importance Score")
    ax.set_title("Top 10 Most Predictive Factors in Churn Decision")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<center style='color: #64748B;'>Built for Data Science Portfolio & Japanese Tech Placement Interviews | Antigravity AI</center>", unsafe_allow_html=True)

