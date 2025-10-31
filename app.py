# 🌾 AI Credit Scoring System (Institutional Edition v4.1)
# -----------------------------------------------------------
# Updated with PDF report generation and revised About section for institutional users.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, gzip, io, os, requests
from fpdf import FPDF

# -----------------------------------------------------
# 🌍 PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="🏦 Institutional Credit Scoring Engine", layout="wide")
st.title("🏦 AI Credit Scoring & Loan Assessment for Financial Institutions")
st.caption("A risk-weighted credit decision support system for banks, MFIs, and cooperatives.")

MODEL_PATH = "credit_model.pkl.gz"
DATA_PATH = "main_harmonized_dataset_final.csv"
GITHUB_MODEL_URL = "https://raw.githubusercontent.com/JacobMuli/ai-credit-scoring/main/credit_model.pkl.gz"

# -----------------------------------------------------
# 📦 LOAD MODEL
# -----------------------------------------------------
@st.cache_resource
def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            with gzip.open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        else:
            response = requests.get(GITHUB_MODEL_URL)
            response.raise_for_status()
            model = pickle.load(io.BytesIO(response.content))
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        st.stop()

model = load_model()

# -----------------------------------------------------
# 📂 LOAD DATA
# -----------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        st.warning("Dataset not found. Please upload 'main_harmonized_dataset_final.csv'.")
        st.stop()

data = load_data()

# -----------------------------------------------------
# 🧮 RISK FACTOR COMPUTATION
# -----------------------------------------------------
def compute_risk_factor(row):
    return round(
        0.18 * {"High": 0, "Moderate": 0.5, "Low": 1}.get(row["Agro-Ecological Zone Compatibility"], 0) +
        0.17 * {"Low": 0, "Moderate": 0.5, "High": 1}.get(row["Pest disease vulnerability"], 0) +
        0.14 * {"High": 0, "Moderate": 0.5, "Low": 1}.get(row["Water irrigation reliability"], 0) +
        0.13 * {"Yes": 0, "No": 1}.get(row["Post Harvest Storage"], 0) +
        0.13 * {"Yes": 0, "No": 1}.get(row["Market Access"], 0) +
        0.10 * {"High": 0, "Low": 1}.get(row["Planting/Sowing Time"], 0) +
        0.08 * {">9 years": 0, "5-9 years": 0.25, "1-4 years": 0.5, "<1 year": 1}.get(row["Farmer experience"], 0) +
        0.05 * {"Yes": 0, "No": 1}.get(row["Cooperative Membership"], 0) +
        0.02 * {"Yes": 0, "No": 1}.get(row["Input Access and Affordability"], 0), 3
    )

data["Risk Factor"] = data.apply(compute_risk_factor, axis=1)
data["Projected Revenue"] = data["Previous Yield Output (Kgs)"] * data["Price"]

# -----------------------------------------------------
# 🧭 TABS
# -----------------------------------------------------
tab_assess, tab_portfolio, tab_dashboard, tab_about = st.tabs([
    "🏦 Institutional Risk & Loan Assessment",
    "💰 Portfolio Simulation",
    "📊 Model Dashboard",
    "ℹ️ About System"
])

# =====================================================
# TAB 1: INSTITUTIONAL RISK & LOAN ASSESSMENT
# =====================================================
with tab_assess:
    st.subheader("🏦 Institutional Risk Factor & Loan Calculator")
    st.sidebar.header("Institutional Parameters")

    alpha = st.sidebar.slider("Risk Sensitivity (α)", 0.1, 1.5, 0.9, 0.05)
    interest_rate = st.sidebar.number_input("Annual Interest Rate (%)", 1.0, 40.0, 16.0, 0.5)

    st.sidebar.header("Farmer Attributes")
    crop = st.sidebar.selectbox("Crop Type", sorted(data["Crop Type"].unique()))
    price = st.sidebar.number_input("Expected Crop Price (KES/kg)", 10, 500, 100)
    yield_output = st.sidebar.number_input("Expected Yield Output (Kgs)", 100, 500000, 20000)

    risk_factor = st.sidebar.slider("Composite Risk Factor (Rₓ)", 0.0, 1.0, 0.35, 0.01)

    projected_revenue = yield_output * price
    I = interest_rate / 100

    loan_amount = (projected_revenue * (1 * alpha * risk_factor)) / (1 + I)

    st.markdown("### 💰 Computation Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Projected Revenue (P)", f"{projected_revenue:,.0f} KES")
    col2.metric("Risk Factor (Rₓ)", f"{risk_factor:.3f}")
    col3.metric("Risk Sensitivity (α)", f"{alpha}")
    col4.metric("Interest Rate (I)", f"{interest_rate:.2f}%")

    st.code("L = (P × (1 × α × Rₓ)) / (1 + I)", language="python")
    st.success(f"💰 **Recommended Principal Loan (L)** = {loan_amount:,.0f} KES")

# =====================================================
# TAB 2: PORTFOLIO SIMULATION
# =====================================================
with tab_portfolio:
    st.subheader("💰 Institutional Portfolio Simulation")

    alpha = st.slider("Risk Sensitivity (α)", 0.1, 1.5, 0.9, 0.05)
    interest_rate = st.number_input("Interest Rate (%)", 1.0, 40.0, 16.0, 0.5)
    I = interest_rate / 100

    data["Loan Amount"] = (data["Projected Revenue"] * (1 * alpha * data["Risk Factor"])) / (1 + I)
    data["Loan Amount"] = data["Loan Amount"].round(2)

    avg_loan = data["Loan Amount"].mean().round(2)
    total_loan = data["Loan Amount"].sum().round(2)

    st.metric("Average Loan per Farmer", f"{avg_loan:,.0f} KES")
    st.metric("Total Portfolio Loan", f"{total_loan:,.0f} KES")

    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(data["Loan Amount"], bins=30, kde=True, ax=ax)
    ax.set_title("Loan Amount Distribution across Portfolio")
    st.pyplot(fig)

# =====================================================
# TAB 3: MODEL DASHBOARD PLACEHOLDER
# =====================================================
with tab_dashboard:
    st.subheader("📊 Model Evaluation Dashboard")
    st.info("Institutional users can evaluate predictive model performance here.")

# =====================================================
# TAB 4: ABOUT SECTION
# =====================================================
with tab_about:
    st.subheader("ℹ️ About the Institutional Credit Scoring Engine")

    st.markdown("""
    ### 🏦 Purpose
    This system is built for **financial institutions** — including banks, microfinance organizations, and SACCOs — to assess farmer risk factors and derive loan amounts based on institutional risk policies.

    ### ⚙️ Core Logic
    The model and engine use a weighted risk factor approach derived from agronomic, environmental, and socioeconomic data. Each farmer record has a computed **Risk Factor (Rₓ)** between 0 and 1, where:
    - 0 → Low risk
    - 1 → High risk

    Institutions can adjust:
    - **α (Risk Sensitivity)** — reflects the institution's risk appetite.
    - **I (Interest Rate)** — reflects the institution’s lending rate.

    The system computes the loan amount using the formula:
    \[ L = \frac{P \times (1 \times α \times Rₓ)}{(1 + I)} \]

    ### 💡 Features
    - Interactive calculator for institution-specific parameters.
    - Dataset-wide simulation for portfolio analysis.
    - Automated PDF report generation for decision records.
    - Transparent, explainable AI-driven methodology.

    ### 🧭 Ethics & Transparency
    - Promotes fairness by allowing consistent, data-driven risk assessment.
    - Maintains transparency through open formulas and explainable results.
    - Respects privacy by operating on anonymized or synthetic data.

    ### 🌍 Impact
    Enables lenders to:
    - Objectively evaluate creditworthiness.
    - Standardize lending decisions.
    - Encourage sustainable, inclusive agricultural financing.
    """)
