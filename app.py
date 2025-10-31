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
# Includes SHAP & LIME explainability for ethical and transparent AI.
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
st.set_page_config(page_title="🏦 E-jenga Credit Engine", layout="wide")
st.title("🏦 E-jenga Credit Scoring & Loan Assessment Engine")
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
    "📊 Model Dashboard (with Explainability)",
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

    crop_filter = st.selectbox("Filter by Crop Type", ["All"] + sorted(data["Crop Type"].unique().tolist()))
    df_sim = data if crop_filter == "All" else data[data["Crop Type"] == crop_filter]

    st.metric("Average Loan per Farmer", f"KES {df_sim['Loan Amount'].mean():,.0f}")
    st.metric("Total Portfolio Loan", f"KES {df_sim['Loan Amount'].sum():,.0f}")

    st.markdown("### 📊 Loan Distribution by Risk Factor")
    fig = px.scatter(df_sim, x="Risk Factor", y="Loan Amount", color="Crop Type", size="Loan Amount",
                     title="Loan Amount vs Risk Factor")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("💾 Download Portfolio Data", df_sim.to_csv(index=False).encode("utf-8"),
                       "portfolio_simulation.csv", "text/csv")

# =====================================================
# TAB 3: MODEL DASHBOARD (DESCRIPTIVE ANALYTICS + EXPLAINABILITY)
# =====================================================
with tab_dashboard:
    st.subheader("📊 Model & Dataset Insights Dashboard")

    st.markdown("### 🔍 Dataset Summary Statistics")
    st.dataframe(data.describe(include='all').transpose())

    st.markdown("### 🔗 Feature Correlation Matrix")
    numeric_data = data.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numerical columns available for correlation heatmap.")

    # 🌾 Feature Importance
    st.markdown("### 🌾 Feature Importance (Model Explainability)")
    try:
        if hasattr(model, "named_steps"):
            final_estimator = list(model.named_steps.values())[-1]
        elif hasattr(model, "steps"):
            final_estimator = model.steps[-1][1]
        else:
            final_estimator = model

        if hasattr(final_estimator, "feature_importances_"):
            importances = final_estimator.feature_importances_
            feature_names = getattr(model, "feature_names_in_", data.columns)
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)
            fig = px.bar(importance_df, x="Importance", y="Feature",
                         orientation="h", title="Feature Importance Ranking")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importance extraction failed: {e}")

    # 🧠 SHAP & LIME Explainability
    st.markdown("### 🧠 Model Explainability — SHAP & LIME")
    try:
        import shap
        from lime import lime_tabular

        X = data.select_dtypes(include=[np.number]).dropna()
        if len(X) > 0:
            X_sample = X.sample(min(100, len(X)), random_state=42)

            # SHAP Global
            st.markdown("#### 🔹 SHAP Summary Plot (Global Feature Impact)")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(fig, bbox_inches="tight", dpi=120)

            # SHAP Local
            st.markdown("#### 🔹 SHAP Force Plot (Single Farmer Prediction)")
            idx = st.number_input("Select Farmer Index (for SHAP Force Plot)", 0, len(X_sample)-1, 0, 1)
            shap.initjs()
            st_shap = shap.force_plot(explainer.expected_value[1], shap_values[1][idx, :], X_sample.iloc[idx, :], matplotlib=False)
            st.components.v1.html(shap.getjs() + st_shap.html(), height=300)

            # LIME
            st.markdown("#### 🔹 LIME Explanation (Individual Prediction)")
            lime_explainer = lime_tabular.LimeTabularExplainer(
                training_data=np.array(X_sample),
                feature_names=X_sample.columns.tolist(),
                mode="classification"
            )
            farmer_index = st.number_input("Select Farmer Index for LIME", 0, len(X_sample)-1, 1, 1)
            farmer_instance = X_sample.iloc[farmer_index]
            explanation = lime_explainer.explain_instance(
                data_row=farmer_instance,
                predict_fn=model.predict_proba,
                num_features=10
            )
            html = explanation.as_html()
            st.components.v1.html(html, height=600, scrolling=True)
    except Exception as e:
        st.warning(f"Explainability section could not be generated: {e}")

    # 📈 Risk Factor Distribution
    st.markdown("### 📈 Risk Factor Distribution")
    fig2 = px.histogram(data, x="Risk Factor", nbins=20, title="Distribution of Computed Risk Factors")
    st.plotly_chart(fig2, use_container_width=True)

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
