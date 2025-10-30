# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="🌾 AI Credit Scoring", layout="centered")

st.title("🌾 AI Credit Scoring for Smallholder Farmers")
st.write("Predict a farmer’s creditworthiness based on basic farm and mobile data.")

MODEL_PATH = "credit_model.pkl"

@st.cache_resource
def load_model(path=MODEL_PATH):
    """Load the trained credit scoring model."""
    if not os.path.exists(path):
        return None, f"❌ Model file not found at {path}. Upload credit_model.pkl to this folder."
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"⚠️ Error loading model: {e}"

model, load_error = load_model()

if load_error:
    st.error(load_error)
    st.stop()

# Sidebar input form
st.sidebar.header("📋 Farmer Profile")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 80, 35)
farm_size = st.sidebar.number_input("Farm size (hectares)", 0.1, 100.0, 3.5, step=0.1)
crop = st.sidebar.selectbox("Main Crop", ["Maize", "Beans", "Tea", "Coffee", "Horticulture"])
cooperative = st.sidebar.selectbox("Member of Cooperative", [0, 1])
yield_hist = st.sidebar.number_input("Average yield (tons/ha)", 0.1, 10.0, 2.5, step=0.1)
mobile_txns = st.sidebar.number_input("Monthly Mobile Transactions", 0, 200, 25)
mobile_balance = st.sidebar.number_input("Avg. Mobile Wallet Balance (KES)", 0, 100000, 1500)
ndvi = st.sidebar.slider("NDVI (Vegetation Health)", 0.05, 0.9, 0.55, step=0.01)
drought_exposure = st.sidebar.selectbox("Drought Exposure (recent)", [0, 1])

sample = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "farm_size": farm_size,
    "crop": crop,
    "cooperative": cooperative,
    "yield_hist": yield_hist,
    "mobile_txns": mobile_txns,
    "mobile_balance": mobile_balance,
    "ndvi": ndvi,
    "drought_exposure": drought_exposure,
}])

st.markdown("---")

if st.button("🚀 Predict Credit Score"):
    try:
        prob_default = model.predict_proba(sample)[0, 1]
        credit_score = (1 - prob_default) * 1000
        eligible = credit_score >= 400

        st.subheader("🔍 Prediction Results")
        st.metric("Credit Score", f"{credit_score:.0f}")
        st.metric("Default Probability", f"{prob_default:.2%}")

        if eligible:
            loan_amount = min(sample["farm_size"].values[0] * 300, 50000)
            interest_rate = 0.12 + prob_default * 0.5
            st.success("✅ Farmer is **eligible** for credit!")
            st.write(f"**Suggested Loan Amount:** KES {loan_amount:,.0f}")
            st.write(f"**Suggested Interest Rate:** {interest_rate*100:.2f}%")
        else:
            st.error("❌ Farmer **not eligible** for loan at this time.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
