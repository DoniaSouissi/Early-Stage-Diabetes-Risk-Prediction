import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk AI",
    page_icon="🩺",
    layout="centered", # 'wide' or 'centered'
    initial_sidebar_state="collapsed"
)

# Custom CSS for "Beautiful Colors" and "Card" styling
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f363d 0%, #5e1c7a 100%);
    }
    
    /* Header Styling */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-text {
        text-align: center;
        color: #7F8C8D;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Card Container for Inputs */
    .input-card {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.4);
    }
    
    /* Result Cards */
    .result-safe {
        background-color: #d4edda;
        color: #155724;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        text-align: center;
    }
    .result-danger {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------
# 2. LOAD MODEL & TOOLS
# ------------------------------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load('models/best_model.joblib')
    preprocessor = joblib.load('models/preprocessor.joblib')
    return model, preprocessor

try:
    model, preprocessor = load_assets()
except FileNotFoundError:
    st.error("⚠️ Model files not found! Please run the notebooks first to generate 'models/' folder.")
    st.stop()

# ------------------------------------------------------------------------------------------------
# 3. UI LAYOUT
# ------------------------------------------------------------------------------------------------

# Header
st.markdown("<h1 class='main-header'>🩺 AI Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Enter patient symptoms below to generate a real-time risk assessment using Random Forest AI.</p>", unsafe_allow_html=True)

# Main Form inside a "Card"
with st.container():
    
    with st.form("prediction_form"):
        st.subheader("📋 Patient Details")
        
        # Row 1: Demographics
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (Years)", min_value=1, max_value=120, value=35)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])

        st.markdown("---")
        st.subheader("🤒 Symptoms Checklist")

        # Organize symptoms into 3 clean columns for better layout
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
    
    st.markdown("</div>", unsafe_allow_html=True) # End of Card

# ------------------------------------------------------------------------------------------------
# 4. PREDICTION LOGIC
# ------------------------------------------------------------------------------------------------
if submit_btn:
    # 1. Map "Yes"/"No" back to original Format if needed, or keep as is.
    # Note: Our preprocessor expects "Yes"/"No" strings exactly like the training data.
    
    # 2. Create DataFrame
    # MUST match the exact column names and order from training!
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
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

    # 3. Process & Predict
    try:
        processed_data = preprocessor.transform(input_data)
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        # 4. Display Result with Custom CSS classes
        st.markdown("### 📊 Assessment Result")
        
        if prediction == 1:
            # High Risk Styling
            st.markdown(f"""
                <div class='result-danger'>
                    <h2>⚠️ High Risk Detected</h2>
                    <p style='font-size: 1.2rem;'>The model predicts a <strong>{probability:.1%}</strong> probability of Early Stage Diabetes.</p>
                    <hr>
                    <p><strong>Recommendation:</strong> Please consult a healthcare professional for a blood glucose test immediately.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Low Risk Styling
            st.markdown(f"""
                <div class='result-safe'>
                    <h2>✅ Low Risk Detected</h2>
                    <p style='font-size: 1.2rem;'>The model predicts a low probability (<strong>{probability:.1%}</strong>) of diabetes.</p>
                    <hr>
                    <p><strong>Recommendation:</strong> Maintain a healthy lifestyle and continue regular checkups.</p>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")