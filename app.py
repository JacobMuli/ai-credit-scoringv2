# 🌾 AI Credit Scoring System (Institutional Edition v5.1 Final)
# -----------------------------------------------------------
# Unified Institutional + Farmer-Level Credit Scoring Engine
# -----------------------------------------------------------
# Combines:
# - Farmer agronomic metrics (AEZ, Pest, Water, Market, etc.)
# - Institutional parameters (α = Risk Sensitivity, I = Interest Rate)
# Implements loan formula:
#   L = (P × (1 × α × Rₓ)) / (1 + I)
# Generates dynamic risk assessment, portfolio simulation, model dashboard, and PDF reports.
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, gzip, io, os, requests
from fpdf import FPDF
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import plotly.express as px

# -----------------------------------------------------
# 🌍 PAGE CONFIGURATION
# -----------------------------------------------------
st.set_page_config(page_title="🏦 Institutional Credit Scoring Engine", layout="wide")
st.title("🏦 AI Credit Scoring & Loan Assessment for Financial Institutions")
st.caption("Institutional decision-support combining farmer agronomic risk with institutional lending policies.")

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
# 🧮 RISK FACTOR COMPUTATION FUNCTION
# -----------------------------------------------------
def compute_risk_from_row(r):
    return round(
        0.18 * {"High": 0, "Moderate": 0.5, "Low": 1}.get(r.get("Agro-Ecological Zone Compatibility", "High"), 0) +
        0.17 * {"Low": 0, "Moderate": 0.5, "High": 1}.get(r.get("Pest disease vulnerability", "Low"), 0) +
        0.14 * {"High": 0, "Moderate": 0.5, "Low": 1}.get(r.get("Water irrigation reliability", "High"), 0) +
        0.13 * {"Yes": 0, "No": 1}.get(r.get("Post Harvest Storage", "Yes"), 0) +
        0.13 * {"Yes": 0, "No": 1}.get(r.get("Market Access", "Yes"), 0) +
        0.10 * {"High": 0, "Low": 1}.get(r.get("Planting/Sowing Time", "High"), 0) +
        0.08 * {">9 years": 0, "5-9 years": 0.25, "1-4 years": 0.5, "<1 year": 1}.get(r.get("Farmer experience", ">9 years"), 0) +
        0.05 * {"Yes": 0, "No": 1}.get(r.get("Cooperative Membership", "Yes"), 0) +
        0.02 * {"Yes": 0, "No": 1}.get(r.get("Input Access and Affordability", "Yes"), 0), 3
    )

# Add missing computed columns
if "Risk Factor" not in data.columns:
    data["Risk Factor"] = data.apply(lambda row: compute_risk_from_row(row), axis=1)

if "Projected Revenue" not in data.columns and "Previous Yield Output (Kgs)" in data.columns:
    data["Projected Revenue"] = data["Previous Yield Output (Kgs)"] * data["Price"]

# -----------------------------------------------------
# 🧭 TABS
# -----------------------------------------------------
tab_assess, tab_portfolio, tab_dashboard, tab_report = st.tabs([
    "🏦 Institutional Risk & Loan Assessment",
    "💰 Portfolio Simulation",
    "📊 Model Dashboard",
    "📄 PDF Report Generator"
])

# =====================================================
# TAB 1: RISK & LOAN CALCULATOR
# =====================================================
with tab_assess:
    st.subheader("🏦 Institutional Risk Factor & Loan Calculator — Farmer Inputs Included")

    st.sidebar.header("Institution Parameters")
    inst_name = st.sidebar.text_input("Institution Name (optional)")
    logo_file = st.sidebar.file_uploader("Upload Institution Logo (optional)", type=["png", "jpg", "jpeg"])
    alpha = st.sidebar.slider("Risk Sensitivity (α)", 0.1, 1.5, 0.9, 0.01)
    interest_rate = st.sidebar.number_input("Annual Interest Rate (%)", 0.0, 100.0, 16.0, 0.1)

    st.sidebar.header("Farmer Inputs")
    aez = st.sidebar.selectbox("Agro-Ecological Zone Compatibility", ["High", "Moderate", "Low"])
    pest = st.sidebar.selectbox("Pest & Disease Vulnerability", ["Low", "Moderate", "High"])
    water = st.sidebar.selectbox("Water & Irrigation Reliability", ["High", "Moderate", "Low"])
    storage = st.sidebar.selectbox("Post-Harvest Storage", ["Yes", "No"])
    market = st.sidebar.selectbox("Market Access", ["Yes", "No"])
    planting = st.sidebar.selectbox("Planting/Sowing Time", ["High", "Low"])
    experience = st.sidebar.selectbox("Farmer Experience", [">9 years", "5-9 years", "1-4 years", "<1 year"])
    coop = st.sidebar.selectbox("Cooperative Membership", ["Yes", "No"])
    input_access = st.sidebar.selectbox("Input Access and Affordability", ["Yes", "No"])

    st.sidebar.header("Economic Inputs")
    price = st.sidebar.number_input("Expected Crop Price (KES/kg)", 1.0, 10000.0, 100.0, 0.1)
    yield_output = st.sidebar.number_input("Expected Yield Output (Kgs)", 1, 1000000, 20000)

    # Compute risk and loan
    risk_factor_calc = compute_risk_from_row({
        "Agro-Ecological Zone Compatibility": aez,
        "Pest disease vulnerability": pest,
        "Water irrigation reliability": water,
        "Post Harvest Storage": storage,
        "Market Access": market,
        "Planting/Sowing Time": planting,
        "Farmer experience": experience,
        "Cooperative Membership": coop,
        "Input Access and Affordability": input_access
    })
    projected_revenue = yield_output * price
    I = interest_rate / 100
    loan_amount = (projected_revenue * (1 * alpha * risk_factor_calc)) / (1 + I)
    eligibility = "✅ Eligible for Financing" if risk_factor_calc <= 0.5 else "⚠️ High Risk - Review Required"

    st.markdown("### 🧮 Computation Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Factor (Rₓ)", f"{risk_factor_calc:.3f}")
    c2.metric("Projected Revenue (P)", f"KES {projected_revenue:,.0f}")
    c3.metric("Risk Sensitivity (α)", f"{alpha}")
    c4.metric("Interest Rate (I)", f"{interest_rate:.2f}%")

    st.success(f"💰 **Recommended Principal Loan (L)** = KES {loan_amount:,.0f}")
    st.info(f"Credit Eligibility: {eligibility}")

# =====================================================
# TAB 2: PORTFOLIO SIMULATION
# =====================================================
with tab_portfolio:
    st.subheader("💰 Institutional Portfolio Simulation")

    alpha_p = st.slider("Institution Risk Sensitivity (α)", 0.1, 1.5, 0.9, 0.01)
    interest_rate_p = st.number_input("Interest Rate (%)", 0.0, 100.0, 16.0, 0.1)
    I_p = interest_rate_p / 100

    data["Loan Amount"] = (data["Projected Revenue"] * (1 * alpha_p * data["Risk Factor"])) / (1 + I_p)
    data["Loan Amount"] = data["Loan Amount"].round(2)

    # Filters
    crop_filter = st.selectbox("Filter by Crop Type", ["All"] + sorted(data["Crop Type"].unique().tolist()))
    df_sim = data if crop_filter == "All" else data[data["Crop Type"] == crop_filter]

    # Portfolio metrics
    st.metric("Average Loan per Farmer", f"KES {df_sim['Loan Amount'].mean():,.0f}")
    st.metric("Total Portfolio Loan", f"KES {df_sim['Loan Amount'].sum():,.0f}")

    st.markdown("### 📊 Loan Distribution by Risk Factor")
    fig = px.scatter(df_sim, x="Risk Factor", y="Loan Amount", color="Crop Type", size="Loan Amount",
                     title="Loan Amount vs Risk Factor")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("💾 Download Portfolio Data", df_sim.to_csv(index=False).encode("utf-8"),
                       "portfolio_simulation.csv", "text/csv")

# =====================================================
# TAB 3: MODEL DASHBOARD
# =====================================================
with tab_dashboard:
    st.subheader("📊 Model Performance Dashboard")
    if "default" in data.columns:
        try:
            X = data.drop(columns=["default"], errors="ignore").select_dtypes(include=["number"])
            y = data["default"]
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_proba)
            st.metric("ROC-AUC Score", f"{auc:.3f}")

            # Confusion Matrix
            cm = confusion_matrix(y, y_pred)
            st.write("### Confusion Matrix")
            st.dataframe(pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]))

            # Classification Report
            report = classification_report(y, y_pred, output_dict=True)
            st.write(pd.DataFrame(report).transpose())

        except Exception as e:
            st.error(f"Error evaluating model: {e}")
    else:
        st.warning("No 'default' column found — unable to compute model metrics.")

# =====================================================
# TAB 4: PDF REPORT GENERATOR
# =====================================================
with tab_report:
    st.subheader("📄 Generate Institutional Loan Report (PDF)")
    if st.button("📄 Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, txt="Institutional Loan Report", ln=True, align="C")
        pdf.ln(10)
        pdf.cell(0, 10, txt=f"Institution: {inst_name}", ln=True)
        pdf.cell(0, 10, txt=f"Risk Sensitivity (α): {alpha}", ln=True)
        pdf.cell(0, 10, txt=f"Interest Rate (I): {interest_rate}%", ln=True)
        pdf.cell(0, 10, txt=f"Risk Factor (Rₓ): {risk_factor_calc}", ln=True)
        pdf.cell(0, 10, txt=f"Loan Amount: KES {loan_amount:,.0f}", ln=True)
        pdf.output("institutional_loan_report.pdf")
        with open("institutional_loan_report.pdf", "rb") as f:
            st.download_button("⬇️ Download Report", f, "institutional_loan_report.pdf")
