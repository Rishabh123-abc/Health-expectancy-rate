import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Health Expectancy Predictor",
    page_icon="🩺",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #f8fbff;
    }

    .title {
        font-size: 38px;
        font-weight: 800;
        color: #0f172a;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #475569;
        text-align: center;
        margin-bottom: 25px;
    }

    .info-box {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #dbeafe;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    .result-box {
        background: linear-gradient(135deg, #dbeafe, #eff6ff);
        padding: 22px;
        border-radius: 18px;
        text-align: center;
        border: 1px solid #bfdbfe;
        box-shadow: 0 6px 18px rgba(59,130,246,0.18);
        margin-top: 20px;
    }

    .result-text {
        font-size: 20px;
        font-weight: 700;
        color: #1e3a8a;
    }

    .small-note {
        color: #64748b;
        font-size: 14px;
        text-align: center;
        margin-top: 8px;
    }

    div.stButton > button {
        width: 100%;
        height: 3.2em;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        font-size: 18px;
        font-weight: 700;
        transition: 0.3s ease-in-out;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #1e40af);
        transform: scale(1.01);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model_data = joblib.load("health_expectancy.joblib")

model = model_data["model"]
scaler = model_data["scaler"]
columns = model_data["columns"]

# ---------------- HEADER ----------------
st.markdown('<div class="title">🩺 Health Expectancy Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict estimated life expectancy using year, healthcare spending, and country.</div>',
    unsafe_allow_html=True
)

# ---------------- INFO BOX ----------------
st.markdown("""
<div class="info-box">
<b>How this works:</b><br>
This model predicts <b>Life Expectancy (in years)</b> based on:
<ul>
<li><b>Year</b></li>
<li><b>Health Spending (USD)</b></li>
<li><b>Country</b></li>
</ul>
Enter the details below and click <b>Predict Life Expectancy</b>.
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
st.subheader("📋 Enter Details")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input(
        "Enter Year",
        min_value=1970,
        max_value=2035,
        value=2020,
        step=1
    )

with col2:
    spending = st.number_input(
        "Health Spending (USD)",
        min_value=0.0,
        value=2000.0,
        step=100.0
    )

country = st.selectbox(
    "Select Country",
    ["Canada", "France", "Germany", "Great Britain", "Japan", "USA"]
)

# ---------------- PREDICTION ----------------
if st.button("Predict Life Expectancy"):

    # Create input dictionary with all columns = 0
    input_dict = {col: 0 for col in columns}

    # Fill numeric values
    if "Year" in input_dict:
        input_dict["Year"] = year

    if "Spending_USD" in input_dict:
        input_dict["Spending_USD"] = spending

    # Set selected country dummy = 1
    country_col = f"Country_{country}"
    if country_col in input_dict:
        input_dict[country_col] = 1

    # Convert to DataFrame in same training column order
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    # Display result
    st.markdown(f"""
        <div class="result-box">
            <div class="result-text">Predicted Life Expectancy: {prediction:.2f} years</div>
        </div>
    """, unsafe_allow_html=True)

    # Extra message
    if prediction < 40 or prediction > 100:
        st.warning("⚠️ Predicted value is outside the usual life expectancy range. Please verify the input values.")
    else:
        st.success("Prediction generated successfully.")

# ---------------- FOOTER ----------------
st.markdown(
    '<div class="small-note">Built with Streamlit • Machine Learning Regression Model</div>',
    unsafe_allow_html=True
)