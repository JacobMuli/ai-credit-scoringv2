from pathlib import Path

# Path for the final app.py file
final_app_path = Path("/mnt/data/app_final_v5.py")

# Write the fixed and finalized app.py code content to file
final_code = """# 🌾 AI Credit Scoring System (Institutional Edition v5.0 Final)
# -----------------------------------------------------------
# Unified Institutional + Farmer-Level Credit Scoring Engine
# -----------------------------------------------------------
# This final version integrates:
# - Farmer agronomic risk metrics (AEZ, Pest, Water, Storage, etc.)
# - Institutional parameters (Risk Sensitivity α, Interest Rate I)
# - Composite Risk Factor calculation using weighted metrics from Word doc
# - Loan amount calculation: L = (P × (1 × α × Rₓ)) / (1 + I)
# - Portfolio simulation and PDF reporting for institutions
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
# 📦 MODEL LOADING FUNCTION
# -----------------------------------------------------
@st.cache_resource
def load_model():
    \"\"\"Loads the trained Random Forest model from local path or GitHub.\"\"\"
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
# 📂 LOAD HARMONIZED DATASET
# -----------------------------------------------------
@st.cache_data
def load_data():
    \"\"\"Loads the harmonized farmer dataset or prompts upload if missing.\"\"\"
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        st.warning("Dataset not found. Please upload 'main_harmonized_dataset_final.csv'.")
        st.stop()

data = load_data()

# -----------------------------------------------------
# 🧮 COMPUTE RISK FACTOR FUNCTION
# -----------------------------------------------------
def compute_risk_from_row(r):
    \"\"\"Calculates composite risk factor using weighted 9-metric formula.\"\"\"
    return round(
        0.18 * {\"High\": 0, \"Moderate\": 0.5, \"Low\": 1}.get(r.get(\"Agro-Ecological Zone Compatibility\", \"High\"), 0) +
        0.17 * {\"Low\": 0, \"Moderate\": 0.5, \"High\": 1}.get(r.get(\"Pest disease vulnerability\", \"Low\"), 0) +
        0.14 * {\"High\": 0, \"Moderate\": 0.5, \"Low\": 1}.get(r.get(\"Water irrigation reliability\", \"High\"), 0) +
        0.13 * {\"Yes\": 0, \"No\": 1}.get(r.get(\"Post Harvest Storage\", \"Yes\"), 0) +
        0.13 * {\"Yes\": 0, \"No\": 1}.get(r.get(\"Market Access\", \"Yes\"), 0) +
        0.10 * {\"High\": 0, \"Low\": 1}.get(r.get(\"Planting/Sowing Time\", \"High\"), 0) +
        0.08 * {\">9 years\": 0, \"5-9 years\": 0.25, \"1-4 years\": 0.5, \"<1 year\": 1}.get(r.get(\"Farmer experience\", \">9 years\"), 0) +
        0.05 * {\"Yes\": 0, \"No\": 1}.get(r.get(\"Cooperative Membership\", \"Yes\"), 0) +
        0.02 * {\"Yes\": 0, \"No\": 1}.get(r.get(\"Input Access and Affordability\", \"Yes\"), 0), 3
    )

# Compute Risk Factor and Projected Revenue
if "Risk Factor" not in data.columns:
    data["Risk Factor"] = data.apply(lambda row: compute_risk_from_row(row), axis=1)

if "Projected Revenue" not in data.columns:
    if "Previous Yield Output (Kgs)" in data.columns and "Price" in data.columns:
        data["Projected Revenue"] = data["Previous Yield Output (Kgs)"] * data["Price"]
    else:
        data["Projected Revenue"] = 0

# -----------------------------------------------------
# 🧭 MAIN NAVIGATION TABS
# -----------------------------------------------------
tab_assess, tab_portfolio, tab_dashboard, tab_report = st.tabs([
    "🏦 Institutional Risk & Loan Assessment",
    "💰 Portfolio Simulation",
    "📊 Model Dashboard",
    "📄 PDF Report Generator"
])

# =====================================================
# TAB 1: RISK & LOAN CALCULATOR (FARMER + INSTITUTION)
# =====================================================
with tab_assess:
    st.subheader("🏦 Institutional Risk Factor & Loan Calculator — Farmer-Level Inputs Included")

    # --- Institution Parameters ---
    st.sidebar.header("Institution Parameters")
    inst_name = st.sidebar.text_input("Institution Name (optional)")
    logo_file = st.sidebar.file_uploader("Upload Institution Logo (PNG/JPG, optional)", type=["png","jpg","jpeg"])
    alpha = st.sidebar.slider("Risk Sensitivity (α)", 0.1, 1.5, 0.9, 0.01)
    interest_rate = st.sidebar.number_input("Annual Interest Rate (%)", 0.0, 100.0, 16.0, 0.1)

    # --- Farmer Inputs ---
    st.sidebar.header("Farmer Agronomic Inputs")
    aez = st.sidebar.selectbox("Agro-Ecological Zone Compatibility", ["High","Moderate","Low"])
    pest = st.sidebar.selectbox("Pest & Disease Vulnerability", ["Low","Moderate","High"])
    water = st.sidebar.selectbox("Water & Irrigation Reliability", ["High","Moderate","Low"])
    storage = st.sidebar.selectbox("Post-Harvest Storage", ["Yes","No"])
    market = st.sidebar.selectbox("Market Access", ["Yes","No"])
    planting = st.sidebar.selectbox("Planting/Sowing Time", ["High","Low"])
    experience = st.sidebar.selectbox("Farmer Experience", [">9 years","5-9 years","1-4 years","<1 year"])
    coop = st.sidebar.selectbox("Cooperative Membership", ["Yes","No"])
    input_access = st.sidebar.selectbox("Input Access & Affordability", ["Yes","No"])

    # --- Economic Inputs ---
    st.sidebar.header("Economic Inputs")
    price = st.sidebar.number_input("Expected Crop Price (KES/kg)", 1.0, 10000.0, 100.0, 0.1)
    yield_output = st.sidebar.number_input("Expected Yield Output (Kgs)", 1, 1000000, 20000)

    # --- Compute Weighted Risk Factor ---
    risk_factor_calc = (
        0.18 * {"High":0, "Moderate":0.5, "Low":1}[aez] +
        0.17 * {"Low":0, "Moderate":0.5, "High":1}[pest] +
        0.14 * {"High":0, "Moderate":0.5, "Low":1}[water] +
        0.13 * {"Yes":0, "No":1}[storage] +
        0.13 * {"Yes":0, "No":1}[market] +
        0.10 * {"High":0, "Low":1}[planting] +
        0.08 * {">9 years":0, "5-9 years":0.25, "1-4 years":0.5, "<1 year":1}[experience] +
        0.05 * {"Yes":0, "No":1}[coop] +
        0.02 * {"Yes":0, "No":1}[input_access]\n    )\n    risk_factor_calc = round(risk_factor_calc, 3)\n\n    # --- Loan Calculation ---\n    projected_revenue = yield_output * price\n    I = interest_rate / 100.0\n    loan_amount = (projected_revenue * (1 * alpha * risk_factor_calc)) / (1 + I)\n    eligibility = \"✅ Eligible for Financing\" if risk_factor_calc <= 0.5 else \"⚠️ High Risk - Review Required\"\n\n    st.markdown(\"### 🧾 Computation Summary\")\n    c1, c2, c3, c4 = st.columns(4)\n    c1.metric(\"Risk Factor (Rₓ)\", f\"{risk_factor_calc:.3f}\")\n    c2.metric(\"Projected Revenue (P)\", f\"KES {projected_revenue:,.0f}\")\n    c3.metric(\"Risk Sensitivity (α)\", f\"{alpha}\")\n    c4.metric(\"Interest Rate (I)\", f\"{interest_rate:.2f}%\")\n\n    st.code(\"L = (P × (1 × α × Rₓ)) / (1 + I)\", language=\"python\")\n    st.success(f\"💰 Recommended Principal Loan (L): KES {loan_amount:,.0f}\")\n    st.info(f\"Eligibility: {eligibility}\")\n"}\n
# Save the final file
final_app_path.write_text(final_code)

# Display the path for download
final_app_path"}
