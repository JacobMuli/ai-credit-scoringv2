# 🌾 AI Credit Scoring for Smallholder Farmers

This Streamlit app predicts a farmer's creditworthiness using demographic, financial, and environmental data.

## 🚀 Features
- Predict credit score & default probability  
- Determine loan eligibility and interest rate  
- Ready for deployment on [Streamlit Cloud](https://share.streamlit.io)

## 🧠 Model
The model (`credit_model.pkl`) is trained using a Random Forest Classifier on synthetic farmer data generated in Google Colab.  
It uses features like:
- Gender, Age, Crop Type  
- Farm Size & Yield History  
- Mobile Money Transactions  
- NDVI (Vegetation Index)  
- Drought Exposure & Cooperative Membership  

## ⚙️ Setup & Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
