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
# 🧭 TABS
# -----------------------------------------------------
tab_predict, tab_financing, tab_dashboard = st.tabs([
    "🧾 Predict Farmer Credit Score",
    "💰 Financing & Loan Simulation",
    "📊 Model Performance Dashboard"
])

# =====================================================
# TAB 1: RISK FACTOR & FINANCING ANALYSIS
# =====================================================
with tab_predict:
    st.subheader("🌿 Risk Factor & Financing Assessment")
    st.sidebar.header("Farmer Profile Inputs")

    # --- User Inputs ---
    crop = st.sidebar.selectbox("Crop Type", sorted(data["Crop Type"].unique()))
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    farm_size = st.sidebar.number_input("Farm Size (hectares)", 0.1, 100.0, 3.0)
    yield_output = st.sidebar.number_input("Previous Yield Output (Kgs)", 100, 500000, 20000)
    price = st.sidebar.number_input("Crop Price (KES/kg)", 10, 500, 100)
    age = st.sidebar.slider("Age", 18, 90, 40)
    coop = st.sidebar.selectbox("Cooperative Membership", ["Yes", "No"])

    # --- Risk Inputs ---
    aez = st.sidebar.selectbox("Agro-Ecological Zone Compatibility", ["High", "Moderate", "Low"])
    pest = st.sidebar.selectbox("Pest & Disease Vulnerability", ["Low", "Moderate", "High"])
    water = st.sidebar.selectbox("Water & Irrigation Reliability", ["High", "Moderate", "Low"])
    storage = st.sidebar.selectbox("Post-Harvest Storage", ["Yes", "No"])
    market = st.sidebar.selectbox("Market Access", ["Yes", "No"])
    planting = st.sidebar.selectbox("Planting/Sowing Time", ["High", "Low"])
    experience = st.sidebar.selectbox("Farmer Experience", [">9 years", "5-9 years", "1-4 years", "<1 year"])
    input_access = st.sidebar.selectbox("Input Access and Affordability", ["Yes", "No"])

    # --- Compute Risk Factor (Weighted Formula) ---
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

    # --- Financing Calculation ---
    projected_revenue = yield_output * price
    loan_amount = projected_revenue * (1 - risk_factor)
    eligibility = "✅ Eligible for Financing" if risk_factor <= 0.5 else "⚠️ High Risk - Not Eligible"

    # --- Display Results ---
    st.markdown("### 🧾 Farmer Risk & Financing Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Agro Risk Factor", f"{risk_factor:.3f}")
    col2.metric("Projected Revenue (KES)", f"{projected_revenue:,.0f}")
    col3.metric("Recommended Loan (KES)", f"{loan_amount:,.0f}")
    col4.metric("Status", eligibility)

    # --- Explanation Section ---
    st.markdown("### 🧮 Calculation Breakdown")
    st.write(f"""
    - **Weighted Risk Factor:** Computed using the official weights from the Risk Factor Formula.
    - **Projected Revenue:** Yield × Price = {yield_output:,.0f} × {price} = {projected_revenue:,.0f} KES
    - **Loan Amount Formula:** Projected Revenue × (1 – Risk Factor) = {projected_revenue:,.0f} × (1 – {risk_factor}) = {loan_amount:,.0f} KES
    - **Eligibility:** Determined by Risk Factor threshold (≤ 0.5 = Eligible)
    """)

# =====================================================
# TAB 2: FINANCING SIMULATION
# =====================================================
with tab_financing:
    st.subheader("💰 Loan and Financing Simulation")
    st.write("Loan amounts are based on risk-adjusted projected revenue.")

    data["Projected Revenue"] = data["Previous Yield Output (Kgs)"] * data["Price"]
    data["Loan Amount"] = (data["Projected Revenue"] * (1 - data["Risk Factor"])).round(2)

    st.dataframe(data[["Farmer ID", "Crop Type", "Risk Factor", "Projected Revenue", "Loan Amount"]].head(10))

    st.markdown("### 📊 Loan Amount Distribution by Crop Type")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x="Crop Type", y="Loan Amount", ax=ax)
    ax.set_title("Loan Distribution by Crop Type")
    st.pyplot(fig)

    # 📤 Export simulated loan data
    st.markdown("### 📤 Export Loan Simulation Data")
    csv_data = data[["Farmer ID", "Crop Type", "Risk Factor", "Projected Revenue", "Loan Amount"]].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Loan Simulation Data as CSV",
        data=csv_data,
        file_name="loan_simulation_results.csv",
        mime="text/csv"
    )

# =====================================================
# TAB 3: MODEL DASHBOARD
# =====================================================
with tab_dashboard:
    st.subheader("📊 Model Performance Dashboard")

    if "default" in data.columns:
        X = data.drop(columns=["default"])
        y_true = data["default"]
        try:
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = roc_auc_score(y_true, y_proba)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.legend(); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            st.pyplot(fig)

            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Model evaluation unavailable: {e}")
