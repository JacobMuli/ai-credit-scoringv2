# 🌾 AI Hackathon Streamlit App (Final Branch-Aware Version)
# -----------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, gzip, io, os, requests
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# -----------------------------------------------------
# 🌍 PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="🌾 AI Credit Scoring", layout="wide")
st.title("🌾 AI Credit Scoring for Smallholder Farmers")
st.caption("An AI-powered simulation built for the Intro to AI 4 Startups Hackathon.")

# -----------------------------------------------------
# 📦 LOAD MODEL (AUTO-DETECT GITHUB BRANCH)
# -----------------------------------------------------
MODEL_PATH = "credit_model.pkl.gz"
DATA_PATH = "main_harmonized_dataset_final.csv"
GITHUB_BASE = "https://raw.github.com/JacobMuli/ai-credit-scoringv2"
DEFAULT_BRANCH = "main"  # fallback if detection fails

def detect_github_branch():
    branch = os.getenv("GIT_BRANCH")
    if branch:
        return branch
    return DEFAULT_BRANCH

def load_model():
    branch = detect_github_branch()
    github_url = f"{GITHUB_BASE}/{branch}/credit_model.pkl.gz"
    try:
        if os.path.exists(MODEL_PATH):
            with gzip.open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            st.success(f"✅ Model loaded locally from {MODEL_PATH}")
        else:
            st.info(f"📥 Loading model from GitHub branch `{branch}` ...")
            response = requests.get(github_url)
            response.raise_for_status()
            model = pickle.load(io.BytesIO(response.content))
            st.success(f"✅ Model loaded from branch `{branch}`")
        return model
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

model = load_model()

# -----------------------------------------------------
# 📂 LOAD DATASET
# -----------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df
    else:
        st.warning("⚠️ Dataset not found. Please upload 'main_harmonized_dataset_final.csv'.")
        st.stop()

data = load_data()

# -----------------------------------------------------
# 🧮 VERIFY OR RECOMPUTE RISK FACTOR
# -----------------------------------------------------
def normalize_values(df):
    df["Computed_Risk_Factor"] = (
        0.18 * df["Agro-Ecological Zone Compatibility"].map({"High": 0, "Moderate": 0.5, "Low": 1}) +
        0.17 * df["Pest disease vulnerability"].map({"Low": 0, "Moderate": 0.5, "High": 1}) +
        0.14 * df["Water irrigation reliability"].map({"High": 0, "Moderate": 0.5, "Low": 1}) +
        0.13 * df["Post Harvest Storage"].map({"Yes": 0, "No": 1}) +
        0.13 * df["Market Access"].map({"Yes": 0, "No": 1}) +
        0.10 * df["Planting/Sowing Time"].map({"High": 0, "Low": 1}) +
        0.08 * df["Farmer experience"].map({">9 years": 0, "5-9 years": 0.25, "1-4 years": 0.5, "<1 year": 1}) +
        0.05 * df["Cooperative Membership"].map({"Yes": 0, "No": 1}) +
        0.02 * df["Input Access and Affordability"].map({"Yes": 0, "No": 1})
    ).round(3)

    tolerance = 0.02
    if "Risk Factor" in df.columns:
        similarity = np.mean(np.isclose(df["Risk Factor"], df["Computed_Risk_Factor"], atol=tolerance))
        if similarity < 0.95:
            df["Risk Factor"] = df["Computed_Risk_Factor"]
    else:
        df["Risk Factor"] = df["Computed_Risk_Factor"]

    df["default"] = (df["Risk Factor"] > 0.5).astype(int)
    return df

data = normalize_values(data)
st.caption(f"📁 Using model from branch: `{detect_github_branch()}`")

# -----------------------------------------------------
# 🌍 SUMMARY DASHBOARD
# -----------------------------------------------------
st.markdown("### 📊 Dataset Overview Dashboard")
total_farmers = len(data)
avg_risk = data["Risk Factor"].mean().round(3)
total_projected_loan = (data["Previous Yield Output (Kgs)"] * data["Price"] * (1 - data["Risk Factor"])).sum()

colA, colB, colC = st.columns(3)
colA.metric("Total Farmers", f"{total_farmers}")
colB.metric("Average Risk Factor", f"{avg_risk}")
colC.metric("Total Projected Loan (KES)", f"{total_projected_loan:,.0f}")

# -----------------------------------------------------
# 🧭 TABS
# -----------------------------------------------------
tab_predict, tab_financing, tab_dashboard, tab_about = st.tabs([
    "🧾 Risk Factor & Financing Analysis",
    "💰 Financing & Loan Simulation",
    "📊 Model Performance Dashboard",
    "ℹ️ About Project"
])

# =====================================================
# TAB 1: RISK FACTOR & FINANCING ANALYSIS (NEW FARMER)
# =====================================================
with tab_predict:
    st.subheader("🌿 New Farmer Risk Factor & Financing Assessment")
    st.sidebar.header("Enter New Farmer Details")

    crop = st.sidebar.selectbox("Crop Type", sorted(data["Crop Type"].unique()))
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    farm_size = st.sidebar.number_input("Farm Size (hectares)", 0.1, 100.0, 3.0)
    yield_output = st.sidebar.number_input("Expected Yield Output (Kgs)", 100, 500000, 20000)
    price = st.sidebar.number_input("Expected Crop Price (KES/kg)", 10, 500, 100)
    age = st.sidebar.slider("Farmer Age", 18, 90, 40)
    coop = st.sidebar.selectbox("Cooperative Membership", ["Yes", "No"])

    aez = st.sidebar.selectbox("Agro-Ecological Zone Compatibility", ["High", "Moderate", "Low"])
    pest = st.sidebar.selectbox("Pest & Disease Vulnerability", ["Low", "Moderate", "High"])
    water = st.sidebar.selectbox("Water & Irrigation Reliability", ["High", "Moderate", "Low"])
    storage = st.sidebar.selectbox("Post-Harvest Storage", ["Yes", "No"])
    market = st.sidebar.selectbox("Market Access", ["Yes", "No"])
    planting = st.sidebar.selectbox("Planting/Sowing Time", ["High", "Low"])
    experience = st.sidebar.selectbox("Farmer Experience", [">9 years", "5-9 years", "1-4 years", "<1 year"])
    input_access = st.sidebar.selectbox("Input Access and Affordability", ["Yes", "No"])

    risk_factor = (
        0.18 * {"High": 0, "Moderate": 0.5, "Low": 1}[aez] +
        0.17 * {"Low": 0, "Moderate": 0.5, "High": 1}[pest] +
        0.14 * {"High": 0, "Moderate": 0.5, "Low": 1}[water] +
        0.13 * {"Yes": 0, "No": 1}[storage] +
        0.13 * {"Yes": 0, "No": 1}[market] +
        0.10 * {"High": 0, "Low": 1}[planting] +
        0.08 * {">9 years": 0, "5-9 years": 0.25, "1-4 years": 0.5, "<1 year": 1}[experience] +
        0.05 * {"Yes": 0, "No": 1}[coop] +
        0.02 * {"Yes": 0, "No": 1}[input_access]
    )
    risk_factor = round(risk_factor, 3)

    projected_revenue = yield_output * price
    loan_amount = projected_revenue * (1 - risk_factor)
    eligibility = "✅ Eligible for Financing" if risk_factor <= 0.5 else "⚠️ High Risk - Not Eligible"

    st.markdown("### 🧾 New Farmer Risk & Financing Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Agro Risk Factor", f"{risk_factor:.3f}")
    col2.metric("Projected Revenue (KES)", f"{projected_revenue:,.0f}")
    col3.metric("Recommended Loan (KES)", f"{loan_amount:,.0f}")
    col4.metric("Status", eligibility)

    st.markdown("### 🎯 Risk Visualization")
    import plotly.graph_objects as go

    gauge_color = "green" if risk_factor < 0.4 else ("orange" if risk_factor <= 0.6 else "red")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_factor,
        title={'text': 'Farmer Risk Factor', 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, 0.4], 'color': 'lightgreen'},
                {'range': [0.4, 0.6], 'color': 'gold'},
                {'range': [0.6, 1], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_factor
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 2: FINANCING SIMULATION (UPDATED)
# =====================================================
with tab_financing:
    st.subheader("💰 Loan and Financing Simulation — Dataset-wide")
    st.write("Loan amounts are computed as: Projected Revenue × (1 - Risk Factor). Use the controls to filter and export results.")

    if "Projected Revenue" not in data.columns:
        data["Projected Revenue"] = data["Previous Yield Output (Kgs)"] * data["Price"]
    data["Loan Amount"] = (data["Projected Revenue"] * (1 - data["Risk Factor"])).round(2)

    colA, colB, colC = st.columns([1,1,1])
    with colA:
        crop_filter = st.selectbox("Filter by Crop Type", options=["All"] + sorted(data["Crop Type"].unique().tolist()))
    with colB:
        risk_band = st.select_slider("Risk band", options=["All","Low (<=0.4)","Moderate (0.4-0.6)","High (>0.6)"], value="All")
    with colC:
        loan_range = st.slider("Loan amount range (KES)", int(data["Loan Amount"].min()), int(max(data["Loan Amount"].max(), 1)), (int(data["Loan Amount"].min()), int(min(data["Loan Amount"].max(), 500000))))

    df_sim = data.copy()
    if crop_filter != "All":
        df_sim = df_sim[df_sim["Crop Type"] == crop_filter]

    if risk_band != "All":
        if "Low" in risk_band:
            df_sim = df_sim[df_sim["Risk Factor"] <= 0.4]
        elif "Moderate" in risk_band:
            df_sim = df_sim[(df_sim["Risk Factor"] > 0.4) & (df_sim["Risk Factor"] <= 0.6)]
        else:
            df_sim = df_sim[df_sim["Risk Factor"] > 0.6]

    df_sim = df_sim[(df_sim["Loan Amount"] >= loan_range[0]) & (df_sim["Loan Amount"] <= loan_range[1])]

    total_farmers = len(df_sim)
    avg_loan = df_sim["Loan Amount"].mean() if total_farmers else 0
    total_loan = df_sim["Loan Amount"].sum() if total_farmers else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Farmers in selection", f"{total_farmers}")
    k2.metric("Average Loan (KES)", f"{avg_loan:,.0f}")
    k3.metric("Total Loan (KES)", f"{total_loan:,.0f}")

    st.markdown("#### 🔎 Sample of simulated loans")
    st.dataframe(df_sim[["Farmer ID","Farm Location","Crop Type","Risk Factor","Projected Revenue","Loan Amount"]].reset_index(drop=True).head(20), use_container_width=True)

    st.markdown("### 📊 Visualizations")
    fig1, ax1 = plt.subplots(figsize=(8,4))
    sns.boxplot(data=df_sim, x="Crop Type", y="Loan Amount", ax=ax1)
    ax1.set_title("Loan Amount Distribution by Crop Type (Filtered)")
    plt.xticks(rotation=30)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8,3))
    df_group = df_sim.groupby("Crop Type")["Loan Amount"].agg(["mean","sum","count"]).sort_values("mean", ascending=False).reset_index()
    sns.barplot(data=df_group, x="Crop Type", y="mean", ax=ax2)
    ax2.set_ylabel("Mean Loan Amount (KES)")
    plt.xticks(rotation=30)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(8,3))
    ax3.hist(df_sim["Risk Factor"], bins=20)
    ax3.set_xlabel("Risk Factor")
    ax3.set_title("Risk Factor Distribution (Filtered)")
    st.pyplot(fig3)

    st.markdown("### 📤 Export filtered simulation results")
    csv_data = df_sim[["Farmer ID","Farm Location","Crop Type","Risk Factor","Projected Revenue","Loan Amount"]].to_csv(index=False).encode("utf-8")
    st.download_button(label="💾 Download filtered simulation CSV", data=csv_data, file_name="filtered_loan_simulation.csv", mime="text/csv")

# =====================================================
# TAB 3: MODEL PERFORMANCE DASHBOARD (UPDATED FOR NEW DATASET)
# =====================================================
with tab_dashboard:
    st.subheader("📊 Model Performance Dashboard — Using Harmonized Dataset")

    if "default" not in data.columns:
        st.warning("No 'default' column available in dataset to evaluate model. Create a binary target first.")
    else:
        expected_features = [
            'Agro-Ecological Zone Compatibility', 'Pest disease vulnerability',
            'Water irrigation reliability', 'Post Harvest Storage', 'Market Access',
            'Planting/Sowing Time', 'Farmer experience', 'Cooperative Membership',
            'Input Access and Affordability', 'Crop Type', 'Gender', 'Farm size',
            'Previous Yield Output (Kgs)', 'Age'
        ]

        present = [c for c in expected_features if c in data.columns]
        missing = [c for c in expected_features if c not in data.columns]

        if missing:
            st.warning(f"Some expected feature columns are missing from dataset and will be filled with defaults: {missing}")

        X = pd.DataFrame()
        for col in expected_features:
            if col in data.columns:
                X[col] = data[col]
            else:
                if col in ["Farm size","Previous Yield Output (Kgs)","Age"]:
                    X[col] = 0
                else:
                    try:
                        X[col] = data[present].iloc[:,0].mode()[0]
                    except Exception:
                        X[col] = "Unknown"

        y_true = data["default"]

        try:
            y_pred

# =====================================================
# TAB 4: ABOUT PROJECT (UPDATED FROM README)
# =====================================================
with tab_about:
    st.subheader("ℹ️ About the AI Credit Scoring Project")

    st.markdown("""
    ### 🌾 Overview
    The **AI Credit Scoring for Smallholder Farmers** system is designed to bridge the financing gap in agriculture by leveraging data-driven risk analysis. Built as part of the **Intro to AI 4 Startups Hackathon**, the project integrates machine learning, risk weighting, and financial simulation to make lending decisions more inclusive and explainable.

    ### 📘 Dataset: Harmonized Risk & Financing Data
    This version uses the **Harmonized Agricultural Risk Dataset (2025)**, a combined and cleaned dataset merging both farmer demographic and environmental risk factors. Each record includes quantitative and qualitative metrics such as:
    - Agro-Ecological Zone (AEZ) Compatibility
    - Pest & Disease Vulnerability
    - Water & Irrigation Reliability
    - Post-Harvest Storage Availability
    - Market Access
    - Farmer Experience and Cooperative Membership
    - Input Access and Affordability

    These metrics are normalized and weighted into a single **Risk Factor (0–1)**, representing the likelihood of default. A lower score means lower risk and higher financing eligibility.

    ### 🧠 Model & Analytics
    The AI model (Random Forest) was trained using features from the harmonized dataset. It predicts the probability of default, validated using metrics like **ROC-AUC**, **confusion matrix**, and **classification accuracy**.

    Beyond predictive analytics, the application also computes:
    - **Weighted Risk Factor per farmer**
    - **Projected Revenue (Yield × Price)**
    - **Loan Recommendation** = Projected Revenue × (1 – Risk Factor)

    ### 💡 Key Features
    - **Interactive Risk Visualization Gauge**: Provides a real-time view of an individual farmer’s risk factor.
    - **Loan Simulation Dashboard**: Simulates financing scenarios across crops and risk levels.
    - **Performance Analytics**: Monitors model accuracy, AUC score, and feature importance.
    - **Report Generator**: Exports a detailed farmer-level PDF including risk and financing calculations.

    ### 🧩 Technology Stack
    - **Python** (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Plotly, Streamlit)
    - **FPDF2** for generating downloadable PDF reports
    - **GitHub & Streamlit Cloud** for version control and web hosting

    ### 👩🏾‍🌾 Impact
    The system empowers microfinance institutions, cooperatives, and agri-lenders to:
    - Objectively assess farmer creditworthiness
    - Simulate lending risk under various environmental and economic factors
    - Support transparent, data-driven credit allocation in the agricultural sector

    ### 🧭 Future Enhancements
    - Integration with **satellite-based weather data** for dynamic AEZ updates
    - Incorporation of **mobile-based farmer feedback** loops
    - Expansion to regional datasets for cross-country model generalization

    ---
    **Developed by:** Jacob Mwalugho Muli  
    **Hackathon:** Intro to AI 4 Startups — Responsible & Inclusive AgriTech Challenge  
    **Version:** 2.0 (Harmonized Dataset Integration)
    """)
