# 🌾 AI Credit Scoring System (Institutional Edition v5.0 Final)
# -----------------------------------------------------------
# Unified Institutional + Farmer-Level Credit Scoring Engine
# -----------------------------------------------------------
# Combines:
# - Farmer agronomic metrics (AEZ, Pest, Water, Market, etc.)
# - Institutional parameters (α = Risk Sensitivity, I = Interest Rate)
# Implements loan formula:
#   L = (P × (1 × α × Rₓ)) / (1 + I)
# Generates dynamic risk assessment, portfolio simulation, and PDF reports.
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, gzip, io, os, requests
from fpdf import FPDF
from PIL import Image

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
# 🧮 RISK FACTOR COMPUTATION
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

# Compute risk and projected revenue if missing
if "Risk Factor" not in data.columns:
    data["Risk Factor"] = data.apply(lambda row: compute_risk_from_row(row), axis=1)

if "Projected Revenue" not in data.columns:
    if "Previous Yield Output (Kgs)" in data.columns and "Price" in data.columns:
        data["Projected Revenue"] = data["Previous Yield Output (Kgs)"] * data["Price"]
    else:
        data["Projected Revenue"] = 0

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

    # Institution parameters
    st.sidebar.header("Institution Parameters")
    inst_name = st.sidebar.text_input("Institution Name (optional)")
    logo_file = st.sidebar.file_uploader("Upload Institution Logo (optional)", type=["png", "jpg", "jpeg"])
    alpha = st.sidebar.slider("Risk Sensitivity (α)", 0.1, 1.5, 0.9, 0.01)
    interest_rate = st.sidebar.number_input("Annual Interest Rate (%)", 0.0, 100.0, 16.0, 0.1)

    # Farmer agronomic inputs
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

    # Economic inputs
    st.sidebar.header("Economic Inputs")
    price = st.sidebar.number_input("Expected Crop Price (KES/kg)", 1.0, 10000.0, 100.0, 0.1)
    yield_output = st.sidebar.number_input("Expected Yield Output (Kgs)", 1, 1000000, 20000)

    # Compute risk factor
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

    # Loan computation
    projected_revenue = yield_output * price
    I = interest_rate / 100
    loan_amount = (projected_revenue * (1 * alpha * risk_factor_calc)) / (1 + I)
    eligibility = "✅ Eligible for Financing" if risk_factor_calc <= 0.5 else "⚠️ High Risk - Review Required"

    st.metric("Risk Factor (Rₓ)", f"{risk_factor_calc:.3f}")
    st.metric("Loan Amount (KES)", f"{loan_amount:,.0f}")
    st.info(f"Eligibility: {eligibility}")

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
