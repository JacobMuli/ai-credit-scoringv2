# app.py
# 🌾 AI Credit Scoring System (Institutional Edition v5.1 Final — Full app.py with SHAP & LIME)
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

# Explainability libs
import shap
from lime.lime_tabular import LimeTabularExplainer

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

# Extract final estimator (if pipeline) for some checks and for SHAP/LIME usage
def get_final_estimator(model_obj):
    try:
        # sklearn Pipeline
        if hasattr(model_obj, "named_steps"):
            # final estimator is last step
            return list(model_obj.named_steps.values())[-1], model_obj
        if hasattr(model_obj, "steps"):
            return model_obj.steps[-1][1], model_obj
        return model_obj, None
    except Exception:
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

if "Risk Factor" not in data.columns:
    data["Risk Factor"] = data.apply(lambda row: compute_risk_from_row(row), axis=1)

if "Projected Revenue" not in data.columns and "Previous Yield Output (Kgs)" in data.columns and "Price" in data.columns:
    data["Projected Revenue"] = data["Previous Yield Output (Kgs)"] * data["Price"]

# -----------------------------------------------------
# SHAP / LIME HELPERS (cached)
# -----------------------------------------------------
@st.cache_resource
def get_shap_explainer(model_obj, X_sample):
    """
    Returns a SHAP explainer depending on model type.
    If model_obj is a pipeline, user should pass the model that accepts raw X (pipeline recommended).
    """
    try:
        # If model is tree-based or has feature_importances_
        cls_name = model_obj.__class__.__name__.lower()
        if hasattr(model_obj, "feature_importances_") or "tree" in cls_name or "randomforest" in cls_name or "xgboost" in cls_name or "lightgbm" in cls_name:
            explainer = shap.TreeExplainer(model_obj)
            return explainer
        # fallback: KernelExplainer (model-agnostic) but expensive
        # sample background
        X_sample_proc = X_sample.sample(n=min(100, len(X_sample)), random_state=42)
        explainer = shap.KernelExplainer(model_obj.predict_proba, X_sample_proc)
        return explainer
    except Exception as e:
        st.warning(f"SHAP explainer init failed: {e}")
        return None

@st.cache_resource
def get_lime_explainer(X_train, feature_names, class_names):
    """
    Build a LIME Tabular explainer for classification.
    """
    try:
        X_np = X_train.values if hasattr(X_train, "values") else np.array(X_train)
        explainer = LimeTabularExplainer(
            X_np,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True
        )
        return explainer
    except Exception as e:
        st.warning(f"LIME explainer init failed: {e}")
        return None

@st.cache_data
def compute_shap_values(explainer, X):
    """
    Returns shap_values computed for X (pandas DataFrame).
    """
    try:
        if explainer is None:
            return None
        # For classifier: shap values will be list-like where index 1 is positive class
        shap_values = explainer.shap_values(X) if hasattr(explainer, "shap_values") else explainer(X)
        return shap_values
    except Exception as e:
        st.warning(f"SHAP compute failed: {e}")
        return None

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

    # Local explainability block for this new single input
    st.markdown("### 🧾 Local Explanation (SHAP & LIME) — Single Farmer Input")
    try:
        # Build X_new row with columns from data (best-effort)
        expected_features = list(data.columns)
        X_new = pd.DataFrame(index=[0])
        for col in expected_features:
            if col == "Agro-Ecological Zone Compatibility":
                X_new[col] = [aez]
            elif col == "Pest disease vulnerability":
                X_new[col] = [pest]
            elif col == "Water irrigation reliability":
                X_new[col] = [water]
            elif col == "Post Harvest Storage":
                X_new[col] = [storage]
            elif col == "Market Access":
                X_new[col] = [market]
            elif col == "Planting/Sowing Time":
                X_new[col] = [planting]
            elif col == "Farmer experience":
                X_new[col] = [experience]
            elif col == "Cooperative Membership":
                X_new[col] = [coop]
            elif col == "Input Access and Affordability":
                X_new[col] = [input_access]
            elif col == "Previous Yield Output (Kgs)":
                X_new[col] = [yield_output]
            elif col == "Price":
                X_new[col] = [price]
            elif col == "Projected Revenue":
                X_new[col] = [projected_revenue]
            else:
                # default to column mode or zero
                try:
                    X_new[col] = [data[col].mode()[0]]
                except Exception:
                    X_new[col] = [0]
        X_new = X_new.fillna(0)

        st.write("Preview of input features used for explanation:")
        st.dataframe(X_new.T, use_container_width=True)

        # Prepare dataset for numeric/features used for explainers
        X_for_shap = data.select_dtypes(include=[np.number]).fillna(0)
        if X_for_shap.empty:
            # if numeric-only is empty, try to transform categorical with one-hot (best-effort)
            X_for_shap = pd.get_dummies(data).fillna(0).sample(n=min(200, len(data)), random_state=1)

        # If model is pipeline, prefer to pass pipeline.predict_proba and raw X to KernelExplainer
        explainer_model_for_shap = pipeline_obj if pipeline_obj is not None else final_estimator

        shap_explainer = get_shap_explainer(explainer_model_for_shap, X_for_shap)
        shap_vals = None
        if shap_explainer is not None:
            try:
                raw_shap = shap_explainer.shap_values(X_new) if hasattr(shap_explainer, "shap_values") else shap_explainer(X_new)
                shap_vals = raw_shap[1] if isinstance(raw_shap, (list, tuple)) and len(raw_shap) > 1 else raw_shap
            except Exception:
                try:
                    shap_res = shap_explainer(X_new)
                    shap_vals = shap_res.values if hasattr(shap_res, "values") else shap_res
                except Exception as e:
                    st.info(f"SHAP compute fallback failed: {e}")
                    shap_vals = None

        if shap_vals is not None:
            st.markdown("#### SHAP — local contributions (waterfall if available)")
            try:
                # Convert to explanation if needed and plot waterfall
                if isinstance(shap_vals, np.ndarray):
                    # build shap.Explanation
                    base_value = shap_explainer.expected_value if hasattr(shap_explainer, "expected_value") else None
                    shap_expl = shap.Explanation(values=np.atleast_2d(shap_vals), base_values=base_value, data=X_new.values, feature_names=X_new.columns.tolist())
                    plt.figure(figsize=(8, 3))
                    shap.plots.waterfall(shap_expl[0], show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()
                else:
                    # fallback generic plot
                    shap.summary_plot(shap_vals, X_new, show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()
            except Exception as e:
                st.info(f"Could not render SHAP waterfall: {e}")
        else:
            st.info("SHAP local explanation not available for this sample.")

        # LIME local
        st.markdown("#### LIME — approximate local feature weights")
        X_for_lime = data.select_dtypes(include=[np.number]).fillna(0)
        if X_for_lime.empty:
            X_for_lime = pd.get_dummies(data).fillna(0).sample(n=min(200, len(data)), random_state=1)

        feature_names = X_for_lime.columns.tolist()
        class_names = ["class_0", "class_1"]
        lime_explainer = get_lime_explainer(X_for_lime, feature_names, class_names)
        if lime_explainer is not None:
            # build X_new_for_lime with same numeric columns
            X_new_for_lime = pd.DataFrame(columns=feature_names, index=[0]).fillna(0)
            for f in feature_names:
                if f in X_new.columns:
                    X_new_for_lime.at[0, f] = X_new.at[0, f]
                else:
                    # keep dataset mode/population value - already filled by zeros
                    pass
            try:
                exp = lime_explainer.explain_instance(X_new_for_lime.values[0], model.predict_proba, num_features=min(10, len(feature_names)))
                lime_list = exp.as_list()
                lime_df = pd.DataFrame(lime_list, columns=["Feature", "Contribution"])
                st.dataframe(lime_df)
            except Exception as e:
                st.info(f"LIME explanation failed: {e}")
        else:
            st.info("LIME explainer init failed or insufficient numeric features.")

    except Exception as e:
        st.info(f"Local explainability block failed: {e}")

# =====================================================
# TAB 2: PORTFOLIO SIMULATION
# =====================================================
with tab_portfolio:
    st.subheader("💰 Institutional Portfolio Simulation")

    alpha_p = st.slider("Institution Risk Sensitivity (α)", 0.1, 1.5, 0.9, 0.01, key="alpha_p")
    interest_rate_p = st.number_input("Interest Rate (%)", 0.0, 100.0, 16.0, 0.1, key="interest_p")
    I_p = interest_rate_p / 100

    # ensure Projected Revenue exists
    if "Projected Revenue" not in data.columns and "Previous Yield Output (Kgs)" in data.columns and "Price" in data.columns:
        data["Projected Revenue"] = data["Previous Yield Output (Kgs)"] * data["Price"]

    data["Loan Amount"] = (data["Projected Revenue"].fillna(0) * (1 * alpha_p * data["Risk Factor"].fillna(0))) / (1 + I_p)
    data["Loan Amount"] = data["Loan Amount"].round(2)

    crop_filter = st.selectbox("Filter by Crop Type", ["All"] + (sorted(data["Crop Type"].unique().tolist()) if "Crop Type" in data.columns else ["Unknown"]))
    df_sim = data if crop_filter == "All" else data[data["Crop Type"] == crop_filter]

    st.metric("Average Loan per Farmer", f"KES {df_sim['Loan Amount'].mean():,.0f}")
    st.metric("Total Portfolio Loan", f"KES {df_sim['Loan Amount'].sum():,.0f}")

    st.markdown("### 📊 Loan Distribution by Risk Factor")
    fig = px.scatter(df_sim, x="Risk Factor", y="Loan Amount", color="Crop Type" if "Crop Type" in df_sim.columns else None, size="Loan Amount", title="Loan Amount vs Risk Factor")
    st.plotly_chart(fig, width='stretch')

    st.download_button("💾 Download Portfolio Data", df_sim.to_csv(index=False).encode("utf-8"), "portfolio_simulation.csv", "text/csv")

# =====================================================
# TAB 3: MODEL DASHBOARD (DESCRIPTIVE ANALYTICS + SHAP)
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

    st.markdown("### 🌾 Feature Importance (Model Explainability) — SHAP + Model Importances")
    try:
        # try direct feature_importances_
        if hasattr(final_estimator, "feature_importances_"):
            importances = final_estimator.feature_importances_
            if hasattr(model, "feature_names_in_"):
                feature_names = model.feature_names_in_
            else:
                # fallback to numeric_data columns
                feature_names = numeric_data.columns.tolist() if not numeric_data.empty else [f"Feature {i+1}" for i in range(len(importances))]
            importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
            fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h", title="Feature Importance (Model)")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Model does not expose feature_importances_. Using SHAP for model-agnostic explainability (may take time).")
            X_for_shap = data.select_dtypes(include=[np.number]).fillna(0)
            if X_for_shap.empty:
                # try using get_dummies fallback
                X_for_shap = pd.get_dummies(data).fillna(0)
            explainer_model = pipeline_obj if pipeline_obj is not None else final_estimator
            shap_explainer = get_shap_explainer(explainer_model, X_for_shap)
            if shap_explainer is not None:
                X_sample = X_for_shap.sample(n=min(500, len(X_for_shap)), random_state=42)
                shap_values = compute_shap_values(shap_explainer, X_sample)
                if shap_values is not None:
                    try:
                        plt.figure(figsize=(8,6))
                        sv = shap_values[1] if isinstance(shap_values, (list, tuple)) and len(shap_values) > 1 else shap_values
                        shap.summary_plot(sv, X_sample, show=False)
                        st.pyplot(plt.gcf())
                        plt.clf()

                        plt.figure(figsize=(6,6))
                        shap.summary_plot(sv, X_sample, plot_type="bar", show=False)
                        st.pyplot(plt.gcf())
                        plt.clf()

                        # Provide download of mean absolute shap importance
                        mean_abs = np.abs(sv).mean(axis=0)
                        feat_names = X_sample.columns.tolist()
                        shap_imp_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
                        st.download_button("💾 Download SHAP importances (CSV)", shap_imp_df.to_csv(index=False).encode('utf-8'), "shap_importances.csv", "text/csv")
                    except Exception as e:
                        st.warning(f"SHAP plotting failed: {e}")
                else:
                    st.info("Could not compute SHAP values on sample.")
            else:
                st.info("SHAP explainer not available for this model.")
    except Exception as e:
        st.warning(f"Feature importance / SHAP block failed: {e}")

    st.markdown("### 📈 Risk Factor Distribution")
    fig2 = px.histogram(data, x="Risk Factor", nbins=20, title="Distribution of Computed Risk Factors")
    st.plotly_chart(fig2, width='stretch')

# =====================================================
# TAB 4: PDF REPORT GENERATOR
# =====================================================
with tab_report:
    st.subheader("📄 Generate Institutional Loan Report (PDF)")
    if st.button("📄 Generate PDF Report"):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, txt="Institutional Loan Report", ln=True, align="C")
            pdf.ln(10)
            pdf.cell(0, 10, txt=f"Institution: {inst_name if inst_name else 'N/A'}", ln=True)
            pdf.cell(0, 10, txt=f"Risk Sensitivity (alpha): {alpha}", ln=True)
            pdf.cell(0, 10, txt=f"Interest Rate (I): {interest_rate}%", ln=True)
            pdf.cell(0, 10, txt=f"Risk Factor (Rx): {risk_factor_calc}", ln=True)
            pdf.cell(0, 10, txt=f"Loan Amount: KES {loan_amount:,.0f}", ln=True)

            pdf.output("institutional_loan_report.pdf")

            with open("institutional_loan_report.pdf", "rb") as f:
                st.download_button("⬇️ Download Report", f, "institutional_loan_report.pdf")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
