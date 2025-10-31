# 🌾 AI Credit Scoring System (Institutional Edition v5.3 — Full Explainability Stable)
# ----------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, gzip, io, os, requests
import plotly.express as px
from fpdf import FPDF
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# -----------------------------------------------------
# 🌍 PAGE CONFIGURATION
# -----------------------------------------------------
st.set_page_config(page_title="🏦 Institutional Credit Scoring Engine", layout="wide")
st.title("🏦 E-jenga Credit Engine")
st.caption("Institutional decision-support combining farmer agronomic risk with institutional lending policies.")

# -----------------------------------------------------
# 🧠 EXPLAINABILITY LIBRARIES (CPU-SAFE)
# -----------------------------------------------------
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU access
    import shap
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    shap = None
    LimeTabularExplainer = None
    st.warning("⚠️ SHAP or LIME unavailable (torch/CUDA conflict). Explainability will be limited.")

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
# TAB 1: FARMER ASSESSMENT + PER-FARMER EXPLAINABILITY
# =====================================================
with tab_assess:
    st.subheader("🏦 Institutional Risk Factor & Loan Calculator — Farmer Inputs Included")

    st.sidebar.header("Institution Parameters")
    inst_name = st.sidebar.text_input("Institution Name (optional)")
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
    price = st.sidebar.number_input("Expected Crop Price (KES/kg)", 1.0, 10000.0, 100.0, 0.1)
    yield_output = st.sidebar.number_input("Expected Yield Output (Kgs)", 1, 1000000, 20000)

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

    # ===========================
    # Per-Farmer Explainability
    # ===========================
    st.markdown("### 🧾 Local Explainability — Individual Farmer Analysis")
    try:
        X_new = pd.DataFrame({
            "Agro-Ecological Zone Compatibility": [aez],
            "Pest disease vulnerability": [pest],
            "Water irrigation reliability": [water],
            "Post Harvest Storage": [storage],
            "Market Access": [market],
            "Planting/Sowing Time": [planting],
            "Farmer experience": [experience],
            "Cooperative Membership": [coop],
            "Input Access and Affordability": [input_access],
            "Previous Yield Output (Kgs)": [yield_output],
            "Price": [price],
            "Projected Revenue": [projected_revenue],
            "Risk Factor": [risk_factor_calc]
        })

        X_encoded = pd.get_dummies(X_new).reindex(columns=pd.get_dummies(data).columns, fill_value=0)

        if shap is not None:
            try:
                X_trans = pipeline_obj.transform(X_encoded) if pipeline_obj and hasattr(pipeline_obj, "transform") else X_encoded
                explainer = shap.Explainer(final_estimator, X_trans)
                shap_values = explainer(X_trans)
                plt.figure(figsize=(8,3))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(plt.gcf())
                plt.clf()
                st.success("✅ SHAP local explanation generated.")
            except Exception as e:
                st.warning(f"SHAP local explanation unavailable: {e}")

        if LimeTabularExplainer is not None:
            try:
                X_train = pd.get_dummies(data.select_dtypes(include=[np.number])).fillna(0)
                feature_names = X_train.columns.tolist()
                lime_explainer = LimeTabularExplainer(
                    X_train.values,
                    feature_names=feature_names,
                    class_names=["LowRisk", "HighRisk"],
                    discretize_continuous=True)
                exp = lime_explainer.explain_instance(X_encoded.values[0], model.predict_proba, num_features=10)
                lime_df = pd.DataFrame(exp.as_list(), columns=["Feature", "Contribution"])
                st.dataframe(lime_df)
                st.success("✅ LIME explanation generated.")
            except Exception as e:
                st.warning(f"LIME local explanation unavailable: {e}")
    except Exception as e:
        st.warning(f"Local explainability failed: {e}")

# =====================================================
# TAB 3: GLOBAL EXPLAINABILITY DASHBOARD
# =====================================================
with tab_dashboard:
    st.subheader("📊 Model & Dataset Insights Dashboard")

    st.markdown("### 🌾 Feature Importance (Global SHAP or Direct Model)")
    try:
        if hasattr(final_estimator, "feature_importances_"):
            importances = final_estimator.feature_importances_
            features = data.select_dtypes(include=[np.number]).columns
            df_imp = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=False)
            fig = px.bar(df_imp, x="Importance", y="Feature", orientation="h", title="Model Feature Importances")
            st.plotly_chart(fig, width='stretch')
        elif shap is not None:
            X = pd.get_dummies(data.select_dtypes(exclude=["object"])).fillna(0)
            X_sample = X.sample(n=min(100, len(X)), random_state=42)
            if pipeline_obj and hasattr(pipeline_obj, "transform"):
                X_trans = pipeline_obj.transform(X_sample)
                explainer = shap.Explainer(final_estimator, X_trans)
                shap_values = explainer(X_trans)
                feat_names = [f"Feature_{i}" for i in range(X_trans.shape[1])]
            else:
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample)
                feat_names = X_sample.columns
            if shap_values.values.shape[1] == len(feat_names):
                mean_abs = np.abs(shap_values.values).mean(axis=0)
                summary_df = pd.DataFrame({"Feature": feat_names, "Mean|SHAP|": mean_abs}).sort_values("Mean|SHAP|", ascending=False)
                fig = px.bar(summary_df, x="Mean|SHAP|", y="Feature", orientation="h", title="Global SHAP Feature Importances")
                st.plotly_chart(fig, width='stretch')
            else:
                st.warning("⚠️ Feature mismatch due to preprocessing.")
    except Exception as e:
        st.error(f"Explainability error: {e}")

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
        pdf.cell(0, 10, txt=f"Risk Factor: {risk_factor_calc}", ln=True)
        pdf.cell(0, 10, txt=f"Loan Amount: KES {loan_amount:,.0f}", ln=True)
        pdf.output("institutional_loan_report.pdf")
        with open("institutional_loan_report.pdf", "rb") as f:
            st.download_button("⬇️ Download Report", f, "institutional_loan_report.pdf")
