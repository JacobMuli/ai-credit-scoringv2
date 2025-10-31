# app.py
# 🌾 AI Credit Scoring System (Institutional Edition v5.1 Final — SHAP/LIME CPU FIXED VERSION)
# -------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, gzip, io, os, requests
from fpdf import FPDF
from PIL import Image
import plotly.express as px
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# Explainability libs
try:
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU access
except Exception:
    shap = None
    LimeTabularExplainer = None
    st.warning("⚠️ SHAP/LIME libraries partially unavailable. Some explainability features disabled.")

# -----------------------------------------------------
# 🌍 PAGE CONFIGURATION
# -----------------------------------------------------
st.set_page_config(page_title="🏦 Institutional Credit Scoring Engine", layout="wide")
st.title("🏦 E-jenga Credit Engine")
st.caption("Institutional decision-support combining farmer agronomic risk with institutional lending policies.")

MODEL_PATH = "credit_model.pkl.gz"
DATA_PATH = "main_harmonized_dataset_final.csv"
GITHUB_MODEL_URL = "https://raw.githubusercontent.com/JacobMuli/ai-credit-scoring/main/credit_model.pkl.gz"

# -----------------------------------------------------
# 📦 LOAD MODEL
# -----------------------------------------------------
@st.cache_resource(show_spinner=False)
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

# Extract final estimator if pipeline
def get_final_estimator(model_obj):
    if hasattr(model_obj, "named_steps"):
        return list(model_obj.named_steps.values())[-1], model_obj
    elif hasattr(model_obj, "steps"):
        return model_obj.steps[-1][1], model_obj
    return model_obj, None

final_estimator, pipeline_obj = get_final_estimator(model)

# -----------------------------------------------------
# 📂 LOAD DATA
# -----------------------------------------------------
@st.cache_data(show_spinner=False)
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

if "Risk Factor" not in data.columns:
    data["Risk Factor"] = data.apply(lambda row: compute_risk_from_row(row), axis=1)

if "Projected Revenue" not in data.columns and "Previous Yield Output (Kgs)" in data.columns and "Price" in data.columns:
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
# TAB 3: MODEL DASHBOARD — FIXED SHAP IMPLEMENTATION
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

    st.markdown("### 🌾 Feature Importance (Model Explainability) — SHAP CPU SAFE MODE")

    try:
        # If model exposes direct feature importances
        if hasattr(final_estimator, "feature_importances_"):
            importances = final_estimator.feature_importances_
            features = numeric_data.columns if not numeric_data.empty else [f"Feature {i}" for i in range(len(importances))]
            df_imp = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=False)
            fig = px.bar(df_imp, x="Importance", y="Feature", orientation="h", title="Direct Feature Importances")
            st.plotly_chart(fig, width='stretch')
        elif shap is not None:
            # Prepare CPU-safe SHAP computation
            X = pd.get_dummies(data.select_dtypes(exclude=["object"])).fillna(0)
            sample_X = X.sample(n=min(100, len(X)), random_state=42)
            try:
                explainer = shap.Explainer(model, sample_X)
                shap_values = explainer(sample_X)
                mean_abs = np.abs(shap_values.values).mean(axis=0)
                summary_df = pd.DataFrame({"Feature": sample_X.columns, "Mean|SHAP|": mean_abs})
                summary_df = summary_df.sort_values("Mean|SHAP|", ascending=False)

                st.success("✅ SHAP computed successfully in CPU-safe mode.")
                fig = px.bar(summary_df, x="Mean|SHAP|", y="Feature", orientation="h", title="Top SHAP Feature Importances")
                st.plotly_chart(fig, width='stretch')
                st.download_button("💾 Download SHAP Importances (CSV)", summary_df.to_csv(index=False).encode("utf-8"), "shap_importances.csv", "text/csv")
            except Exception as e:
                st.warning(f"SHAP computation failed gracefully: {e}")
        else:
            st.info("SHAP not installed or unavailable. Install with `pip install shap==0.41.0` for CPU-safe mode.")
    except Exception as e:
        st.error(f"Explainability error: {e}")

    st.markdown("### 📈 Risk Factor Distribution")
    fig2 = px.histogram(data, x="Risk Factor", nbins=20, title="Distribution of Computed Risk Factors")
    st.plotly_chart(fig2, width='stretch')

# =====================================================
# PDF GENERATION TAB (unchanged)
# =====================================================
with tab_report:
    st.subheader("📄 Generate Institutional Loan Report (PDF)")
    if st.button("📄 Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, txt="Institutional Loan Report", ln=True, align="C")
        pdf.ln(10)
        pdf.cell(0, 10, txt=f"Total Farmers: {len(data)}", ln=True)
        pdf.cell(0, 10, txt="Generated via E-jenga Credit Engine", ln=True)
        pdf.output("institutional_loan_report.pdf")

        with open("institutional_loan_report.pdf", "rb") as f:
            st.download_button("⬇️ Download Report", f, "institutional_loan_report.pdf")
