import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="DiabRisk AI Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS with Dynamic Background, Particle Effects, and Futuristic Medical Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Roboto:wght@300;400;700&display=swap');
    
    /* ========== RESET & BASE ========== */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* ========== DYNAMIC NEURAL NETWORK BACKGROUND ========== */
    .stApp {
        font-family: 'Roboto', sans-serif !important;
        background: radial-gradient(circle at center, #0a192f 0%, #001b3a 100%) !important;
        position: relative;
        overflow: hidden;
        min-height: 100vh;
        color: #e0f7fa !important;
    }
    
    .stApp::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600"><defs><filter id="glow"><feGaussianBlur stdDeviation="3.5" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><g filter="url(#glow)"><line x1="100" y1="300" x2="500" y2="300" stroke="#00bfff" stroke-width="2" opacity="0.3"><animate attributeName="x1" values="0;600" dur="10s" repeatCount="indefinite"/><animate attributeName="x2" values="0;600" dur="10s" repeatCount="indefinite"/></line><line x1="300" y1="100" x2="300" y2="500" stroke="#00bfff" stroke-width="2" opacity="0.3"><animate attributeName="y1" values="0;600" dur="8s" repeatCount="indefinite"/><animate attributeName="y2" values="0;600" dur="8s" repeatCount="indefinite"/></line><circle cx="150" cy="150" r="5" fill="#00ffff" opacity="0.5"><animate attributeName="cx" values="100;500" dur="15s" repeatCount="indefinite"/><animate attributeName="cy" values="100;500" dur="15s" repeatCount="indefinite"/></circle><circle cx="450" cy="450" r="5" fill="#00ffff" opacity="0.5"><animate attributeName="cx" values="500;100" dur="12s" repeatCount="indefinite"/><animate attributeName="cy" values="500;100" dur="12s" repeatCount="indefinite"/></circle></g></svg>') repeat;
        background-size: 600px 600px;
        animation: neuralFlow 30s linear infinite;
        opacity: 0.15;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes neuralFlow {
        0% { background-position: 0 0; }
        100% { background-position: 600px 600px; }
    }
    
    /* ========== FLOATING MEDICAL PARTICLES ========== */
    .stApp::after {
        content: '🧬 💉 ❤️ ⚕️ 🔬 💊 🩸 📈';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        font-size: 2.5rem;
        opacity: 0.08;
        pointer-events: none;
        word-spacing: 120px;
        line-height: 200px;
        animation: particleFloat 25s linear infinite;
        z-index: -1;
    }
    
    @keyframes particleFloat {
        0% { transform: translateY(100vh) rotate(0deg); }
        100% { transform: translateY(-100vh) rotate(720deg); }
    }
    
    /* ========== MAIN CONTENT CONTAINER ========== */
    .block-container {
        padding: 3rem 2rem !important;
        max-width: 1400px !important;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }
    
    /* ========== FUTURISTIC HEADER ========== */
    .main-header {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin: 1.5rem 0 1rem 0;
        background: linear-gradient(135deg, #00ffff 0%, #00bfff 50%, #1e90ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: neonPulse 2s ease-in-out infinite;
        text-shadow: 0 0 10px rgba(0, 191, 255, 0.5);
        position: relative;
        z-index: 10;
    }
    
    @keyframes neonPulse {
        0%, 100% { filter: brightness(1); text-shadow: 0 0 10px rgba(0, 191, 255, 0.3); }
        50% { filter: brightness(1.5); text-shadow: 0 0 20px rgba(0, 191, 255, 0.7); }
    }
    
    .sub-text {
        text-align: center;
        color: #a0d8ef !important;
        font-size: 1.2rem;
        margin-bottom: 2.5rem;
        font-weight: 300;
        letter-spacing: 1px;
        position: relative;
        z-index: 10;
    }
    
    /* ========== HOLOGRAPHIC CARD ========== */
    div[data-testid="stForm"] {
        background: rgba(10, 25, 47, 0.85) !important;
        backdrop-filter: blur(25px) saturate(200%);
        -webkit-backdrop-filter: blur(25px) saturate(200%);
        border-radius: 30px !important;
        border: 1px solid rgba(0, 191, 255, 0.2) !important;
        padding: 3rem !important;
        box-shadow: 
            0 10px 40px rgba(0, 191, 255, 0.3),
            inset 0 2px 0 rgba(0, 255, 255, 0.1) !important;
        animation: holographicFade 1s ease-out;
        position: relative;
        z-index: 10;
        margin: 1.5rem 0;
    }
    
    @keyframes holographicFade {
        from { 
            opacity: 0; 
            transform: translateY(50px) scale(0.95);
        }
        to { 
            opacity: 1; 
            transform: translateY(0) scale(1);
        }
    }
    
    /* ========== SECTION HEADERS WITH SCAN EFFECT ========== */
    div[data-testid="stForm"] h3 {
        color: #00ffff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 500 !important;
        font-size: 1.5rem !important;
        margin: 2rem 0 1.2rem 0 !important;
        padding-left: 1.2rem;
        border-left: 5px solid #00bfff;
        position: relative;
        overflow: hidden;
    }
    
    div[data-testid="stForm"] h3::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 191, 255, 0.3), transparent);
        animation: scanLine 3s linear infinite;
    }
    
    @keyframes scanLine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* ========== INPUT LABELS ========== */
    .stSelectbox label, .stNumberInput label {
        color: #a0d8ef !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        margin-bottom: 0.6rem !important;
    }
    
    /* Number Input Styling */
    input[type="number"] {
        background: rgba(0, 191, 255, 0.1) !important;
        border: 1px solid rgba(0, 191, 255, 0.4) !important;
        border-radius: 12px !important;
        color: #e0f7fa !important;
        padding: 0.8rem !important;
        transition: all 0.4s ease !important;
        font-size: 1.1rem !important;
    }
    
    input[type="number"]:focus {
        border-color: #00ffff !important;
        box-shadow: 0 0 0 4px rgba(0, 191, 255, 0.2) !important;
        outline: none !important;
    }
    
    /* Dropdown Styling */
    div[data-baseweb="select"] > div {
        background: rgba(0, 191, 255, 0.1) !important;
        border: 1px solid rgba(0, 191, 255, 0.4) !important;
        border-radius: 12px !important;
        color: #e0f7fa !important;
        transition: all 0.4s ease !important;
    }
    
    div[data-baseweb="select"] > div:hover {
        border-color: #00ffff !important;
        box-shadow: 0 0 10px rgba(0, 191, 255, 0.3) !important;
    }
    
    [role="listbox"] {
        background: #0a192f !important;
        border: 1px solid #00bfff !important;
    }
    
    [role="option"] {
        color: #e0f7fa !important;
    }
    
    [role="option"]:hover {
        background: rgba(0, 191, 255, 0.2) !important;
    }
    
    /* ========== SUBMIT BUTTON WITH ENERGY EFFECT ========== */
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #00ffff 0%, #00bfff 100%) !important;
        color: #0a192f !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        border: none !important;
        padding: 1.2rem !important;
        border-radius: 20px !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 2.5rem !important;
        transition: all 0.4s ease !important;
        box-shadow: 0 10px 30px rgba(0, 191, 255, 0.5) !important;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.4s;
    }
    
    .stButton > button:hover::after {
        opacity: 1;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 15px 50px rgba(0, 191, 255, 0.7) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-2px) !important;
    }
    
    /* ========== RESULT CARDS WITH HOLOGRAM EFFECT ========== */
    .result-card {
        padding: 2.5rem;
        border-radius: 25px;
        text-align: center;
        margin-top: 3rem;
        animation: hologramAppear 0.8s ease-out;
        position: relative;
        z-index: 10;
        overflow: hidden;
    }
    
    @keyframes hologramAppear {
        from {
            opacity: 0;
            transform: scale(0.8) translateY(30px);
            filter: blur(5px);
        }
        to {
            opacity: 1;
            transform: scale(1) translateY(0);
            filter: blur(0);
        }
    }
    
    .result-safe {
        background: linear-gradient(135deg, rgba(0, 206, 158, 0.3) 0%, rgba(0, 158, 115, 0.4) 100%);
        border: 2px solid rgba(0, 206, 158, 0.7);
        box-shadow: 0 15px 50px rgba(0, 206, 158, 0.4);
        color: #d4fc79 !important;
    }
    
    .result-danger {
        background: linear-gradient(135deg, rgba(255, 69, 58, 0.3) 0%, rgba(255, 159, 10, 0.4) 100%);
        border: 2px solid rgba(255, 69, 58, 0.7);
        box-shadow: 0 15px 50px rgba(255, 69, 58, 0.4);
        color: #ffd700 !important;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(transparent, rgba(255,255,255,0.1), transparent);
        transform: skewY(45deg);
        animation: hologramScan 2s linear infinite;
        opacity: 0.5;
    }
    
    @keyframes hologramScan {
        0% { top: -100%; }
        100% { top: 100%; }
    }
    
    .result-card h2 {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        color: inherit !important;
        text-shadow: 0 0 8px currentColor;
    }
    
    .result-card p {
        font-size: 1.1rem;
        line-height: 1.8;
        color: inherit !important;
    }
    
    /* ========== PROBABILITY BADGE WITH PULSE ========== */
    .probability-badge {
        display: inline-block;
        padding: 0.8rem 2rem;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50px;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 1.2rem 0;
        backdrop-filter: blur(15px);
        animation: pulseBadge 1.5s ease-in-out infinite;
    }
    
    @keyframes pulseBadge {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* ========== DIVIDER WITH ENERGY FLOW ========== */
    div[data-testid="stForm"] hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ffff, transparent);
        margin: 2.5rem 0;
        animation: energyFlow 2s linear infinite;
    }
    
    @keyframes energyFlow {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    .result-card hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        margin: 1.8rem 0;
    }
    
    /* ========== COLUMNS ========== */
    div[data-testid="column"] {
        padding: 0 1rem !important;
    }
    
    /* ========== FLOATING EMOJIS ========== */
    .emoji-float {
        font-size: 3rem;
        animation: emojiLevitate 4s ease-in-out infinite;
        display: inline-block;
        margin: 0 0.8rem;
    }
    
    @keyframes emojiLevitate {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-15px) rotate(5deg); }
    }
    
    /* ========== SPINNER WITH TECH EFFECT ========== */
    .stSpinner > div {
        border-color: #00ffff transparent transparent transparent !important;
        animation: spinnerGlow 1s linear infinite;
    }
    
    @keyframes spinnerGlow {
        0% { box-shadow: 0 0 5px #00ffff; }
        50% { box-shadow: 0 0 15px #00ffff; }
        100% { box-shadow: 0 0 5px #00ffff; }
    }
    
    /* ========== ERROR & INFO MESSAGES ========== */
    .stAlert {
        background: rgba(10, 25, 47, 0.9) !important;
        border-radius: 20px !important;
        border-left: 5px solid #00ffff !important;
        color: #e0f7fa !important;
        box-shadow: 0 5px 20px rgba(0, 191, 255, 0.2);
    }
    
    /* ========== HIDE STREAMLIT ELEMENTS ========== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* ========== DISCLAIMER BOX WITH SUBTLE GLOW ========== */
    .disclaimer-box {
        text-align: center;
        margin-top: 4rem;
        padding: 2rem;
        background: rgba(0, 191, 255, 0.05);
        border-radius: 20px;
        border: 1px solid rgba(0, 191, 255, 0.3);
        position: relative;
        z-index: 10;
        box-shadow: 0 0 15px rgba(0, 191, 255, 0.1);
    }
    
    .disclaimer-box p {
        font-size: 0.9rem;
        color: #a0d8ef !important;
        line-height: 1.7;
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

# Futuristic Header
st.markdown("<h1 class='main-header'><span class='emoji-float'>🔬</span> DiabRisk AI Pro <span class='emoji-float'>🧬</span></h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>🚀 Futuristic AI for Early Diabetes Detection – Powered by Neural Insights</p>", unsafe_allow_html=True)

# Main Form with Enhanced Layout
with st.form("prediction_form"):
    
    # Section 1: Patient Demographics
    st.markdown("### 👤 Patient Info")
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=40, help="Enter patient's age for risk calibration")
    
    st.markdown("---")
    
    # Section 2: Clinical Symptoms
    st.markdown("### 📋 Symptoms Checklist")

    # Enhanced 3-Column Layout with Icons
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**💧 Hydration Metrics**")
        polyuria = st.selectbox("Polyuria 💦", ["No", "Yes"], help="Excessive urination detected?")
        polydipsia = st.selectbox("Polydipsia 🥤", ["No", "Yes"], help="Unquenchable thirst levels")
        polyphagia = st.selectbox("Polyphagia 🍽️", ["No", "Yes"], help="Abnormal hunger spikes")
        weight_loss = st.selectbox("Weight Loss ⚖️", ["No", "Yes"], help="Sudden mass reduction")
        obesity = st.selectbox("Obesity 📏", ["No", "Yes"], help="BMI overload >30")

    with c2:
        st.markdown("**🧠 Neural System Check**")
        weakness = st.selectbox("Weakness 😴", ["No", "Yes"], help="Energy depletion")
        visual_blurring = st.selectbox("Visual Blurring 👀", ["No", "Yes"], help="Optical distortion")
        irritability = st.selectbox("Irritability 😠", ["No", "Yes"], help="Emotional volatility")
        partial_paresis = st.selectbox("Partial Paresis 💪", ["No", "Yes"], help="Muscle power loss")
        muscle_stiffness = st.selectbox("Muscle Stiffness 🏋️", ["No", "Yes"], help="Rigidity in motion")

    with c3:
        st.markdown("**🩹 Bio-Surface Scan**")
        genital_thrush = st.selectbox("Genital Thrush 🦠", ["No", "Yes"], help="Infection alert")
        itching = st.selectbox("Itching 🐜", ["No", "Yes"], help="Surface irritation")
        delayed_healing = st.selectbox("Delayed Healing ⏳", ["No", "Yes"], help="Recovery slowdown")
        alopecia = st.selectbox("Alopecia 💇", ["No", "Yes"], help="Hair loss")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Submit Button
    submit_btn = st.form_submit_button("🔍 Analyze Risk Now")

# ------------------------------------------------------------------------------------------------
# 4. PREDICTION LOGIC
# ------------------------------------------------------------------------------------------------
if submit_btn:
    with st.spinner('⚡ Neural Processing Activated...'):
        # Create DataFrame
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

        try:
            # Transform and predict
            processed_data = preprocessor.transform(input_data)
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]

            # Display Results
            st.markdown("### 📊 Assessment Result")
            
            if prediction == 1:
                # High Risk Result
                st.markdown(f"""
                    <div class='result-card result-danger'>
                        <h2>🚨 Critical Risk Alert</h2>
                        <div class='probability-badge'>🔴 {probability:.1%} Threat Level</div>
                        <p style='font-size: 1.2rem; margin: 1.8rem 0;'>
                            AI Neural Net detects <strong>high-probability diabetes signature</strong> in bio-markers.
                        </p>
                        <hr>
                        <p style='font-size: 1.1rem;'>
                            <strong>🛑 Defense Protocol:</strong><br>
                            Initiate medical interface: Glucose scan, HbA1c analysis, and expert consultation required.
                        </p>
                        <p style='margin-top: 1.2rem; font-size: 1rem; opacity: 0.9;'>
                            ⚡ Early intervention activates optimal health matrix.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Low Risk Result
                st.markdown(f"""
                    <div class='result-card result-safe'>
                        <h2>🟢 System Stable</h2>
                        <div class='probability-badge'>🟢 {probability:.1%} Threat Level</div>
                        <p style='font-size: 1.2rem; margin: 1.8rem 0;'>
                            Bio-scan shows <strong>minimal diabetes vector</strong> in current profile.
                        </p>
                        <hr>
                        <p style='font-size: 1.1rem;'>
                            <strong>🛡️ Maintenance Protocol:</strong><br>
                            • Optimize nutrition and mobility algorithms<br>
                            • Schedule routine system diagnostics<br>
                            • Enhance hydration and vitality subroutines
                        </p>
                        
                    </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Neural Error: {str(e)}")
            st.info("🔧 Debug Tip: Verify preprocessor and model integrity.")
