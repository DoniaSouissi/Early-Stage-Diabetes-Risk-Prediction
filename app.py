import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="DiabRisk AI",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Gradient Background + Professional Cards
st.markdown("""
    <style>
    /* 1. Main Background - Your Custom Gradient */
    .stApp {
        background: linear-gradient(135deg, #1a484d 0%, #0c663e 100%);
        background-attachment: fixed; /* Keeps gradient fixed while scrolling */
        color: #FAFAFA;
    }
    
    /* 2. Header Styling */
    .main-header {
        font-family: 'Segoe UI', sans-serif;
        color: #ffffff;
        text-align: center;
        font-weight: 700;
        margin-top: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .sub-text {
        text-align: center;
        color: #d1d1d1;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* 3. Card Container (Semi-transparent Dark Box) */
    div[data-testid="stForm"] {
        background-color: rgba(3, 30, 30, 0.8); /* Dark semi-transparent */
        backdrop-filter: blur(20px); /* Glassmorphism effect */
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* 4. Input Fields Styling */
    .stSelectbox label, .stNumberInput label {
        color: #E0E0E0 !important;
        font-weight: 600;
    }
    
    /* Make dropdowns readable */
    div[data-baseweb="select"] > div {
        background-color: #2b2b2b;
        color: white;
    }

    /* 5. Primary Button - Green/Teal Gradient to match theme */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.8rem;
        border-radius: 8px;
        font-size: 1rem;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(56, 239, 125, 0.6);
    }
    
    /* 6. Result Cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .result-safe {
        background-color: rgba(27, 94, 32, 0.9); /* Dark Green */
        color: #e8f5e9;
        border-left: 6px solid #66bb6a;
    }
    .result-danger {
        background-color: rgba(183, 28, 28, 0.9); /* Dark Red */
        color: #ffebee;
        border-left: 6px solid #ef5350;
    }
    
    /* 7. Disclaimer Text */
    .disclaimer {
        font-size: 0.8rem;
        color: #b0bec5;
        text-align: center;
        margin-top: 2rem;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------
# 2. LOAD MODEL & TOOLS
# ------------------------------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/best_model.joblib')
        preprocessor = joblib.load('models/preprocessor.joblib')
        return model, preprocessor
    except Exception as e:
        return None, None

model, preprocessor = load_assets()

if model is None:
    st.error("⚠️ System Error: Model files not found. Please run the training notebooks first.")
    st.stop()

# ------------------------------------------------------------------------------------------------
# 3. UI LAYOUT
# ------------------------------------------------------------------------------------------------

# Header
st.markdown("<h1 class='main-header'>🩺 AI Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Enter patient vitals and symptoms below for a real-time AI assessment.</p>", unsafe_allow_html=True)

# Main Form
with st.form("prediction_form"):
    
    # Section 1: Demographics
    st.markdown("### 👤 Patient Info")
    # Only Age is needed (Gender removed to avoid bias)
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=40)
    
    st.markdown("---")
    
    # Section 2: Clinical Symptoms
    st.markdown("### 📋 Symptoms Checklist")

    # 3-Column Layout
    c1, c2, c3 = st.columns(3)
    
    with c1:
        polyuria = st.selectbox("Polyuria (Excess Urination)", ["No", "Yes"])
        polydipsia = st.selectbox("Polydipsia (Excess Thirst)", ["No", "Yes"])
        weight_loss = st.selectbox("Sudden Weight Loss", ["No", "Yes"])
        weakness = st.selectbox("Weakness", ["No", "Yes"])
        polyphagia = st.selectbox("Polyphagia (Excess Hunger)", ["No", "Yes"])

    with c2:
        genital_thrush = st.selectbox("Genital Thrush", ["No", "Yes"])
        visual_blurring = st.selectbox("Visual Blurring", ["No", "Yes"])
        itching = st.selectbox("Itching", ["No", "Yes"])
        irritability = st.selectbox("Irritability", ["No", "Yes"])
        delayed_healing = st.selectbox("Delayed Healing", ["No", "Yes"])

    with c3:
        partial_paresis = st.selectbox("Partial Paresis", ["No", "Yes"])
        muscle_stiffness = st.selectbox("Muscle Stiffness", ["No", "Yes"])
        alopecia = st.selectbox("Alopecia (Hair Loss)", ["No", "Yes"])
        obesity = st.selectbox("Obesity", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Submit Button
    submit_btn = st.form_submit_button("🔍 Analyze Risk Now")

# ------------------------------------------------------------------------------------------------
# 4. PREDICTION LOGIC
# ------------------------------------------------------------------------------------------------
if submit_btn:
    # 1. Create DataFrame
    # 🚨 CRITICAL: 'Gender' is removed to match your new unbiased model
    input_data = pd.DataFrame({
        'Age': [age],
        'Polyuria': [polyuria],
        'Polydipsia': [polydipsia],
        'sudden weight loss': [weight_loss],
        'weakness': [weakness],
        'Polyphagia': [polyphagia],
        'Genital thrush': [genital_thrush],
        'visual blurring': [visual_blurring],
        'Itching': [itching],
        'Irritability': [irritability],
        'delayed healing': [delayed_healing],
        'partial paresis': [partial_paresis],
        'muscle stiffness': [muscle_stiffness],
        'Alopecia': [alopecia],
        'Obesity': [obesity]
    })

    # 2. Process & Predict
    try:
        # Transform data
        processed_data = preprocessor.transform(input_data)
        
        # Get Prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        # 3. Display Result
        st.markdown("### 📊 Assessment Result")
        
        if prediction == 1:
            # High Risk Styling
            st.markdown(f"""
                <div class='result-card result-danger'>
                    <h2>⚠️ POSITIVE (High Risk)</h2>
                    <p style='font-size: 1.1rem;'>The model predicts a <strong>{probability:.1%}</strong> probability of Diabetes.</p>
                    <hr style='border-color: rgba(255,255,255,0.3);'>
                    <p><strong>Recommendation:</strong> Please consult a doctor immediately.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Low Risk Styling
            st.markdown(f"""
                <div class='result-card result-safe'>
                    <h2>✅ NEGATIVE (Low Risk)</h2>
                    <p style='font-size: 1.1rem;'>The model predicts a <strong>{probability:.1%}</strong> probability (Low Risk).</p>
                    <hr style='border-color: rgba(255,255,255,0.3);'>
                    <p><strong>Recommendation:</strong> Keep up the healthy lifestyle.</p>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Processing Error: {str(e)}")
        st.info("Tip: Ensure you have re-run your '02_data_preparation' notebook to save the new preprocessor without Gender.")
