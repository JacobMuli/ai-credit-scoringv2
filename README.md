🌾 AI Credit Scoring System for Smallholder Farmers (v5.1)
🏦 Institutional & Farmer-Level Risk Assessment Engine

Built for the Intro to AI 4 Startups Hackathon — Agri-Finance, Parametric Insurance & AI-Powered Credit Scoring (2025).
This system empowers financial institutions to assess smallholder farmers’ creditworthiness through AI-driven risk modeling and data-informed lending simulations.

🚀 Project Overview
Component	Description
Track 1 (Model)	Predictive model trained using Random Forests to estimate risk of default using farmer, environmental, and economic data.
Track 2 (Simulation Engine)	Institutional Streamlit app that transforms model outputs into actionable lending decisions based on institutional risk sensitivity (α) and interest rate (I).
Final Output	Interactive, explainable dashboard for loan recommendation, portfolio simulation, and PDF report generation.
🧭 Repository Structure
📦 ai-credit-scoring/
├── src/
│   ├── AI_Hackaton_notebook.ipynb       # Model training, feature engineering, evaluation
│   ├── data_preprocessing.ipynb         # Synthetic data generation & harmonization
│   ├── risk_factor_analysis.ipynb       # Implements weighted risk computation (Word doc formula)
│   └── utils.py                         # Reusable helper functions for normalization, encoding, etc.
│
├── tests/
│   ├── test_model_accuracy.ipynb        # Validation notebook with ROC-AUC, precision, recall
│   ├── test_streamlit_logic.ipynb       # Unit checks for loan formula, normalization, etc.
│   └── test_data_integrity.ipynb        # Confirms dataset completeness & column consistency
│
├── app.py                               # Streamlit institutional interface (v5.1)
├── main_harmonized_dataset_final.csv    # Final harmonized dataset
├── credit_model.pkl.gz                  # Trained ML model (compressed)
├── requirements.txt                     # Dependencies for model & app
├── RISK FACTOR AND FINANCING CALCULATION updated.docx # Documentation of formulas
└── README.md                            # This file

💻 Installation & Setup
1. Clone Repository
git clone https://github.com/JacobMuli/ai-credit-scoring.git
cd ai-credit-scoring

2. Install Dependencies
pip install -r requirements.txt

3. Run the Streamlit Application
streamlit run app.py

🧮 Core Formulas
Composite Risk Factor (Rₓ)

Weighted combination of nine farmer metrics:
Rx​=i∑​(wi​×mi​)

| Metric                                   | Weight | Description                       |
| ---------------------------------------- | ------ | --------------------------------- |
| Agro-Ecological Zone (AEZ) Compatibility | 18%    | Zone suitability for crop         |
| Pest & Disease Vulnerability             | 17%    | Likelihood of infestation         |
| Water & Irrigation Reliability           | 14%    | Irrigation & rainfall consistency |
| Post-Harvest Storage                     | 13%    | Storage capability                |
| Market Access                            | 13%    | Access to produce markets         |
| Planting/Sowing Timing                   | 10%    | Alignment with seasonal calendar  |
| Farmer Experience                        | 8%     | Years of experience               |
| Cooperative Membership                   | 5%     | Access to shared resources        |
| Input Access & Affordability             | 2%     | Availability of inputs            |
​
Loan Financing Formula

L=P×(1×α×Rx​)​/(1+I)

Where:

P = Projected Revenue (Yield × Price)

α = Risk Sensitivity (institutional control)

Rₓ = Weighted Risk Factor (0–1 scale)

I = Interest Rate (in decimal form)

📊 Features (App v5.1)
Feature	Description
🧮 Risk Calculator	Computes farmer risk based on 9 agronomic inputs
🏦 Institutional Controls	Adjustable α (risk sensitivity) and I (interest rate)
💰 Loan Engine	Calculates recommended loan based on institutional formula
📉 Portfolio Simulation	Aggregates risk and loan results across entire dataset
📄 PDF Report Generator	Exports a summarized institutional report with all key inputs
📊 Model Dashboard	Displays ROC-AUC, confusion matrix, and performance overview
📘 Dataset

File: main_harmonized_dataset_final.csv

Combined and harmonized from multiple farmer and environmental datasets.

Includes both qualitative and quantitative risk features.

Synthetic values were generated via Faker and validated manually.

🧠 Model Summary

Algorithm: Random Forest Classifier

Framework: Scikit-learn

Evaluation Metrics: ROC-AUC, Precision, Recall, F1-Score

Explainability: Feature importance and SHAP analysis planned (Track 1 extension)

⚙️ Requirements
Package	Minimum Version
Streamlit	1.20
Scikit-learn	1.0
Pandas	1.3
Numpy	1.21
Seaborn / Matplotlib	Optional (for charts)

(Listed in requirements.txt)

⚖️ Ethical Considerations

Fairness: Avoid bias based on gender or location through feature normalization.

Transparency: Each metric and formula is openly documented.

Accountability: The system provides recommendations, not final decisions.

Inclusivity: Enables underserved farmers’ inclusion into formal finance ecosystems.

🧩 Hackathon Deliverables Alignment
Deliverable	Description
✅ Model Notebook	AI_Hackaton_notebook.ipynb (Track 1)
✅ Loan Simulation Engine	app.py v5.1 (Track 2)
✅ Documentation	RISK FACTOR AND FINANCING CALCULATION updated.docx
✅ Technical Narrative	Current README + embedded comments
✅ Presentation Deck	Available upon request (E-Jenga Team)
👩🏾‍💻 Contributors

Team: E-Jenga
