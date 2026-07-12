import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="S.S_AI | Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — PREMIUM DARK UI
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 10% 0%, #10182b 0%, #0a0e1a 45%, #05070d 100%);
        color: #E6EAF2;
    }

    /* Hide default streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ---------------- BRAND HEADER ---------------- */
    .brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(0,212,255,0.12), rgba(124,58,237,0.14));
        border: 1px solid rgba(0,212,255,0.25);
        box-shadow: 0 8px 32px rgba(0,212,255,0.08);
        margin-bottom: 24px;
        animation: fadeInDown 0.8s ease;
    }
    .brand-logo {
        font-weight: 800;
        font-size: 26px;
        letter-spacing: 1px;
        background: linear-gradient(90deg, #00D4FF, #7C3AED, #00FFA3);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
    }
    .brand-sub {
        font-size: 12.5px;
        color: #8FA1C7;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    .brand-badge {
        background: linear-gradient(90deg, #00D4FF, #7C3AED);
        padding: 8px 18px;
        border-radius: 30px;
        font-size: 12.5px;
        font-weight: 600;
        color: white;
        box-shadow: 0 4px 18px rgba(124,58,237,0.45);
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 0 rgba(0,212,255,0.35); }
        70% { box-shadow: 0 0 0 14px rgba(0,212,255,0); }
        100% { box-shadow: 0 0 0 0 rgba(0,212,255,0); }
    }

    /* ---------------- HERO TITLE ---------------- */
    .hero-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 4px;
        color: #F2F5FA;
    }
    .hero-sub {
        font-size: 15px;
        color: #8FA1C7;
        margin-bottom: 20px;
    }

    /* ---------------- GLASS CARDS ---------------- */
    .glass-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 22px 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        animation: fadeInUp 0.7s ease;
        margin-bottom: 18px;
    }

    .metric-card {
        background: linear-gradient(145deg, rgba(0,212,255,0.08), rgba(124,58,237,0.10));
        border: 1px solid rgba(0,212,255,0.22);
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 26px rgba(0,212,255,0.18);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        background: linear-gradient(90deg, #00D4FF, #00FFA3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 12px;
        color: #8FA1C7;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 4px;
    }

    /* ---------------- RESULT BOXES ---------------- */
    .result-danger {
        background: linear-gradient(135deg, rgba(255,59,92,0.18), rgba(255,59,92,0.06));
        border: 1.5px solid rgba(255,59,92,0.55);
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        animation: pulseGlow 2.2s infinite;
    }
    .result-safe {
        background: linear-gradient(135deg, rgba(0,255,163,0.16), rgba(0,255,163,0.05));
        border: 1.5px solid rgba(0,255,163,0.5);
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        animation: pulseGlow 2.2s infinite;
    }
    .result-title-danger {
        font-size: 24px;
        font-weight: 700;
        color: #FF3B5C;
        margin-bottom: 6px;
    }
    .result-title-safe {
        font-size: 24px;
        font-weight: 700;
        color: #00FFA3;
        margin-bottom: 6px;
    }
    .result-desc {
        color: #C3CEE3;
        font-size: 14px;
    }

    /* ---------------- SIDEBAR ---------------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1424 0%, #080c16 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label {
        color: #A9B8D6 !important;
        font-weight: 500;
        font-size: 13.5px;
    }

    /* ---------------- BUTTON ---------------- */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00D4FF, #7C3AED);
        color: white;
        font-weight: 700;
        font-size: 16px;
        padding: 12px 0;
        border-radius: 14px;
        border: none;
        box-shadow: 0 6px 20px rgba(124,58,237,0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 10px 28px rgba(0,212,255,0.4);
        color: white;
    }

    /* Section labels */
    .section-label {
        font-size: 13px;
        font-weight: 700;
        color: #00D4FF;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        margin: 6px 0 10px 2px;
        border-left: 3px solid #00D4FF;
        padding-left: 10px;
    }

    /* Footer */
    .footer-note {
        text-align: center;
        color: #5C6B8A;
        font-size: 12px;
        margin-top: 30px;
        padding: 14px;
        border-top: 1px solid rgba(255,255,255,0.06);
    }

    hr {
        border-color: rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# BRAND HEADER
# ============================================================
st.markdown("""
<div class="brand-header">
    <div>
        <div class="brand-logo">S.S_AI</div>
        <div class="brand-sub">Intelligent Health Diagnostics</div>
    </div>
    <div class="brand-badge">⚡ Powered by Random Forest ML</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🫀 Heart Disease Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Enter patient clinical data on the left panel to generate an instant AI-powered risk assessment.</div>', unsafe_allow_html=True)

# ============================================================
# DATA LOADING + PREPROCESSING + MODEL TRAINING (CACHED)
# ============================================================
@st.cache_resource
def load_and_train():
    df = pd.read_csv("heart_disease_uci.csv")

    df = df.drop(['id', 'dataset'], axis=1)
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop('num', axis=1)

    df['fbs'] = df['fbs'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
    df['exang'] = df['exang'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})

    numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    categorical_cols = ['sex', 'cp', 'restecg', 'slope', 'thal']

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    df['fbs'] = df['fbs'].fillna(df['fbs'].mode()[0])
    df['exang'] = df['exang'].fillna(df['exang'].mode()[0])

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    return model, scaler, encoders, X.columns.tolist(), acc, auc


with st.spinner("Loading model..."):
    model, scaler, encoders, feature_order, model_acc, model_auc = load_and_train()

# ============================================================
# MODEL PERFORMANCE STRIP
# ============================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{model_acc*100:.1f}%</div>
    <div class="metric-label">Model Accuracy</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{model_auc:.3f}</div>
    <div class="metric-label">ROC-AUC Score</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">200</div>
    <div class="metric-label">Decision Trees</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">920</div>
    <div class="metric-label">Patient Records</div></div>""", unsafe_allow_html=True)

st.write("")

# ============================================================
# SIDEBAR — PATIENT INPUT FORM
# ============================================================
st.sidebar.markdown('<div class="brand-logo" style="font-size:20px;">S.S_AI</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="brand-sub" style="margin-bottom:20px;">Patient Data Entry</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
age = st.sidebar.slider("Age", 18, 100, 50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

st.sidebar.markdown('<div class="section-label">Clinical Symptoms</div>', unsafe_allow_html=True)
cp = st.sidebar.selectbox("Chest Pain Type", [
    "typical angina", "atypical angina", "non-anginal", "asymptomatic"
])
exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])

st.sidebar.markdown('<div class="section-label">Vitals</div>', unsafe_allow_html=True)
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 220, 130)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 240)
thalch = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

st.sidebar.markdown('<div class="section-label">Test Results</div>', unsafe_allow_html=True)
restecg = st.sidebar.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"])
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST", ["upsloping", "flat", "downsloping"])
ca = st.sidebar.slider("Major Vessels Colored (0-3)", 0, 3, 0)
thal = st.sidebar.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

predict_btn = st.sidebar.button("🔬 Run Diagnosis")

# ============================================================
# PREDICTION LOGIC
# ============================================================
def safe_encode(col_name, value):
    """Encode a categorical value, falling back to the most common class if unseen."""
    le = encoders[col_name]
    if value in le.classes_:
        return le.transform([value])[0]
    return le.transform([le.classes_[0]])[0]

if predict_btn:
    input_dict = {
        'age': age,
        'sex': safe_encode('sex', sex),
        'cp': safe_encode('cp', cp),
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs == "Yes" else 0,
        'restecg': safe_encode('restecg', restecg),
        'thalch': thalch,
        'exang': 1 if exang == "Yes" else 0,
        'oldpeak': oldpeak,
        'slope': safe_encode('slope', slope),
        'ca': ca,
        'thal': safe_encode('thal', thal),
    }

    input_df = pd.DataFrame([input_dict])[feature_order]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.markdown('<div class="section-label">Diagnosis Result</div>', unsafe_allow_html=True)

    result_col, gauge_col = st.columns([1.3, 1])

    with result_col:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-danger">
                <div style="font-size:44px;">⚠️</div>
                <div class="result-title-danger">High Risk of Heart Disease</div>
                <div class="result-desc">The model predicts a <b>{probability*100:.1f}%</b> probability of heart disease based on the provided clinical data.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div style="font-size:44px;">✅</div>
                <div class="result-title-safe">Low Risk of Heart Disease</div>
                <div class="result-desc">The model predicts only a <b>{probability*100:.1f}%</b> probability of heart disease based on the provided clinical data.</div>
            </div>
            """, unsafe_allow_html=True)

    with gauge_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Risk Probability**")
        st.progress(float(probability))
        st.markdown(f"<div style='text-align:center; font-size:22px; font-weight:700; margin-top:8px;'>{probability*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 View Full Patient Data Summary"):
        summary_df = pd.DataFrame({
            "Field": ["Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
                      "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate",
                      "Exercise Angina", "ST Depression", "ST Slope", "Major Vessels", "Thalassemia"],
            "Value": [age, sex, cp, f"{trestbps} mm Hg", f"{chol} mg/dl", fbs, restecg,
                      thalch, exang, oldpeak, slope, ca, thal]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.caption("⚕️ This tool is for educational and portfolio purposes only and is not a substitute for professional medical diagnosis.")

else:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:50px 20px;">
        <div style="font-size:40px; margin-bottom:10px;">🩺</div>
        <div style="font-size:18px; font-weight:600; color:#E6EAF2;">Fill in patient details in the sidebar</div>
        <div style="font-size:13.5px; color:#8FA1C7; margin-top:6px;">Click <b>"Run Diagnosis"</b> to generate an AI-powered heart disease risk report.</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer-note">
    Built with ❤️ by <b>S.S_AI</b> &nbsp;|&nbsp; CodeAlpha Machine Learning Internship &nbsp;|&nbsp; Task 4: Disease Prediction
</div>
""", unsafe_allow_html=True)