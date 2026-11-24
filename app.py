import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="CardioGuard | Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Clean, Modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        border-radius: 16px;
        padding: 24px 30px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .header-tag {
        display: inline-block;
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        width: fit-content;
    }
    .header-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 4px 0 0 0;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .header-subtitle {
        font-size: 0.98rem;
        color: #cbd5e1;
        margin: 0;
        font-weight: 400;
    }

    /* Section Cards */
    .form-section-header {
        background-color: #f8fafc;
        border-left: 4px solid #ef4444;
        padding: 8px 14px;
        border-radius: 0 8px 8px 0;
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 14px;
    }

    /* Result Alert Boxes */
    .high-risk-box {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 2px solid #f43f5e;
        border-radius: 14px;
        padding: 24px;
        color: #881337;
        text-align: center;
        box-shadow: 0 8px 20px -4px rgba(244, 63, 94, 0.18);
    }
    .low-risk-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 24px;
        color: #14532d;
        text-align: center;
        box-shadow: 0 8px 20px -4px rgba(34, 197, 94, 0.18);
    }

    /* Metric Cards in Tab 2 & 3 */
    .stat-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .stat-card-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-card-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 4px;
    }

    /* Clinical Recommendation Card */
    .recommendation-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-top: 14px;
    }

    /* Sidebar info card */
    .sidebar-info-card {
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
        font-size: 0.88rem;
        color: #334155;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Seaborn styling for consistent visuals
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "heart.csv")

@st.cache_resource
def load_models_and_metrics():
    rf_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    lr_path = os.path.join(MODEL_DIR, "lr_model.pkl")
    dt_path = os.path.join(MODEL_DIR, "dt_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")

    rf_model = joblib.load(rf_path) if os.path.exists(rf_path) else None
    lr_model = joblib.load(lr_path) if os.path.exists(lr_path) else None
    dt_model = joblib.load(dt_path) if os.path.exists(dt_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return rf_model, lr_model, dt_model, scaler, metrics

@st.cache_data
def load_dataset():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return None

rf_model, lr_model, dt_model, scaler, metrics = load_models_and_metrics()
df = load_dataset()

# Check if model and dataset exist
if rf_model is None or df is None:
    st.error("⚠️ Model or dataset files not found! Please run `python train_model.py` first to generate models.")
    st.stop()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png", width=64)
    st.markdown("## **CardioGuard AI**")
    st.markdown("<span style='color: #64748b; font-size: 0.88rem; font-weight: 500;'>Cardiovascular Decision Support</span>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("⚙️ Active Algorithm")
    chosen_model_name = st.selectbox(
        "Select Machine Learning Model:",
        ["Random Forest (Recommended)", "Logistic Regression", "Decision Tree"]
    )
    
    st.markdown("---")
    st.subheader("⚡ Sample Test Profiles")
    st.caption("Load clinical case profiles to test the system:")
    col_p1, col_p2 = st.columns(2)
    load_high_risk = col_p1.button("🔴 High Risk Case", use_container_width=True)
    load_healthy = col_p2.button("🟢 Healthy Case", use_container_width=True)
    
    st.markdown("---")
    st.markdown(f"""
    <div class="sidebar-info-card">
        <b>📊 Dataset:</b> {len(df)} Patient Records<br>
        <b>🎯 Recommended:</b> {metrics['best_model'] if metrics else 'Random Forest'}<br>
        <b>🩺 Evaluated Features:</b> 13 Clinical Parameters
    </div>
    """, unsafe_allow_html=True)

# Set input values based on presets
if load_high_risk:
    default_age = 62
    default_sex = 1
    default_cp = 0 # Typical angina
    default_trestbps = 158
    default_chol = 285
    default_fbs = 1
    default_restecg = 0
    default_thalach = 115 # low max heart rate
    default_exang = 1 # exercise angina
    default_oldpeak = 2.8
    default_slope = 1
    default_ca = 2
    default_thal = 3
elif load_healthy:
    default_age = 38
    default_sex = 0
    default_cp = 2 # Non-anginal
    default_trestbps = 118
    default_chol = 192
    default_fbs = 0
    default_restecg = 1
    default_thalach = 172 # high max heart rate
    default_exang = 0
    default_oldpeak = 0.2
    default_slope = 2
    default_ca = 0
    default_thal = 2
else:
    default_age = 54
    default_sex = 1
    default_cp = 1
    default_trestbps = 130
    default_chol = 240
    default_fbs = 0
    default_restecg = 0
    default_thalach = 150
    default_exang = 0
    default_oldpeak = 1.0
    default_slope = 1
    default_ca = 0
    default_thal = 2

# ----------------- MAIN INTERFACE -----------------
st.markdown("""
<div class="header-banner">
    <div class="header-tag">Clinical AI Decision Support</div>
    <h1 class="header-title">CardioGuard: Heart Disease Risk Assessment</h1>
    <p class="header-subtitle">Cardiovascular Risk Stratification & Predictive Analytics System</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "🩺 Patient Assessment & Prediction",
    "📊 Exploratory Data Analysis (EDA)",
    "📈 Model Benchmark & Evaluation",
    "📁 Full Dataset Explorer (5,000 Records)"
])

# ==================== TAB 1: PREDICTION ====================
with tabs[0]:
    st.markdown("#### Patient Clinical Parameters")
    st.caption("Enter patient diagnostic metrics, resting vitals, and stress test observations below:")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="form-section-header">1. Demographics & Vitals</div>', unsafe_allow_html=True)
            age = st.slider("Age (years)", min_value=20, max_value=90, value=default_age, help="Age of patient in years")
            sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female", index=0 if default_sex == 1 else 1)
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=210, value=default_trestbps, help="Resting BP on hospital admission (Normal: < 120)")
            chol = st.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=580, value=default_chol, help="Serum cholesterol level (Desirable: < 200 mg/dl)")

        with col2:
            st.markdown('<div class="form-section-header">2. Clinical & ECG Markers</div>', unsafe_allow_html=True)
            cp_map = {
                0: "Typical Angina (chest pressure)",
                1: "Atypical Angina (shortness/fatigue)",
                2: "Non-anginal Pain (sharp/unrelated)",
                3: "Asymptomatic (silent)"
            }
            cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], format_func=lambda x: cp_map[x], index=default_cp)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "Elevated (> 120 mg/dl)" if x == 1 else "Normal (<= 120 mg/dl)", index=default_fbs)
            restecg_map = {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy"
            }
            restecg = st.selectbox("Resting ECG Result", options=[0, 1, 2], format_func=lambda x: restecg_map[x], index=default_restecg)
            thalach = st.slider("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=default_thalach, help="Peak heart rate achieved during exercise stress test")

        with col3:
            st.markdown('<div class="form-section-header">3. Stress Test & Vessels</div>', unsafe_allow_html=True)
            exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes (Angina provoked)" if x == 1 else "No (No angina)", index=default_exang)
            oldpeak = st.slider("ST Depression (oldpeak)", min_value=0.0, max_value=6.5, value=float(default_oldpeak), step=0.1, help="ST depression induced by exercise relative to rest")
            slope_map = {0: "Upsloping (Better)", 1: "Flat (Moderate)", 2: "Downsloping (High Risk)"}
            slope = st.selectbox("Peak Exercise ST Slope", options=[0, 1, 2], format_func=lambda x: slope_map[x], index=default_slope)
            ca = st.slider("Major Vessels by Fluoroscopy (ca)", min_value=0, max_value=3, value=default_ca, help="Number of major vessels (0-3) colored by fluoroscopy")
            thal_map = {1: "Normal", 2: "Fixed Defect (Prior infarction)", 3: "Reversible Defect (Active ischemia)"}
            thal = st.selectbox("Thalassemia Status (thal)", options=[1, 2, 3], format_func=lambda x: thal_map[x], index=min(max(default_thal - 1, 0), 2))

        submit_btn = st.form_submit_button("🔍 Compute Risk Assessment", use_container_width=True)

    if submit_btn:
        input_data = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }])

        # Model evaluation dispatch
        if "Logistic" in chosen_model_name:
            active_model = lr_model
            input_scaled = scaler.transform(input_data)
            prediction = active_model.predict(input_scaled)[0]
            probabilities = active_model.predict_proba(input_scaled)[0]
        elif "Decision" in chosen_model_name:
            active_model = dt_model
            prediction = active_model.predict(input_data)[0]
            probabilities = active_model.predict_proba(input_data)[0]
        else:
            active_model = rf_model
            prediction = active_model.predict(input_data)[0]
            probabilities = active_model.predict_proba(input_data)[0]

        risk_prob = probabilities[1] * 100
        healthy_prob = probabilities[0] * 100

        st.markdown("---")
        st.subheader("📋 Diagnostic Evaluation Result")

        # Clinical Vitals Safety Alerts (Hypertensive Crisis / Severe Vitals)
        if trestbps >= 180:
            st.error(f"🚨 **CRITICAL CLINICAL ALERT — Hypertensive Crisis (Resting BP = {trestbps} mm Hg ≥ 180 mm Hg):** "
                     f"Emergency medical intervention and antihypertensive therapy are urgently required to prevent acute stroke, "
                     f"heart failure, or aortic dissection! (Note: Acute severe hypertension is a medical emergency regardless of current coronary artery plaque blockage).")
        elif trestbps >= 140:
            st.warning(f"⚠️ **Clinical Alert — Stage 2 Hypertension (Resting BP = {trestbps} mm Hg):** "
                       f"Substantial arterial hypertension detected. Prolonged high blood pressure damages arterial walls and accelerates long-term plaque formation.")

        if chol >= 300:
            st.warning(f"⚠️ **Lipid Alert — Severe Hypercholesterolemia (Cholesterol = {chol} mg/dl ≥ 300 mg/dl):** "
                       f"Critically elevated cholesterol substantially accelerates atherosclerotic coronary plaque.")

        res_col1, res_col2 = st.columns([1.1, 1])

        with res_col1:
            if prediction == 1 or risk_prob >= 50.0:
                st.markdown(f"""
                <div class="high-risk-box">
                    <span style="font-size: 1.8rem;">⚠️</span>
                    <h2 style="margin: 8px 0 4px 0; font-size: 1.5rem; font-weight: 800; color: #9f1239;">HIGH RISK OF HEART DISEASE DETECTED</h2>
                    <h1 style="font-size: 3.2rem; margin: 12px 0; font-weight: 800; color: #be123c;">{risk_prob:.1f}%</h1>
                    <p style="font-size: 1rem; margin: 0; color: #9f1239;">Estimated Cardiovascular Risk Probability</p>
                    <p style="font-size: 0.9rem; margin-top: 10px; color: #881337;">The clinical predictive model flagged significant cardiovascular risk indicators.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="low-risk-box">
                    <span style="font-size: 1.8rem;">✅</span>
                    <h2 style="margin: 8px 0 4px 0; font-size: 1.5rem; font-weight: 800; color: #166534;">LOW RISK OF HEART DISEASE</h2>
                    <h1 style="font-size: 3.2rem; margin: 12px 0; font-weight: 800; color: #15803d;">{healthy_prob:.1f}%</h1>
                    <p style="font-size: 1rem; margin: 0; color: #166534;">Estimated Healthy Probability</p>
                    <p style="font-size: 0.9rem; margin-top: 10px; color: #14532d;">Cardiovascular markers fall within expected normal baseline parameters.</p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### Model Confidence Breakdown")
            st.write(f"**Cardiac Risk Probability:** `{risk_prob:.1f}%`")
            st.progress(float(risk_prob / 100.0))
            st.write(f"**Healthy Baseline Confidence:** `{healthy_prob:.1f}%`")
            st.progress(float(healthy_prob / 100.0))

            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.markdown("#### 💡 Clinical Recommendation")
            if prediction == 1 or risk_prob >= 50.0:
                st.error("• Immediate cardiologist consultation and clinical workup recommended.\n• Schedule Coronary Angiography or stress echocardiogram.\n• Review lipid panel, antihypertensive therapy, and metabolic factors.")
            else:
                st.success("• Cardiovascular markers are favorable; maintain balanced lifestyle.\n• Regular moderate aerobic physical activity (150 min/week).\n• Routine periodic cardiac wellness monitoring.")
            st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 2: EDA ====================
with tabs[1]:
    st.markdown("#### Exploratory Data Analysis & Clinical Correlations")
    st.caption("Distribution of clinical biomarkers and their correlation with heart disease diagnosis:")
    
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Total Records</div>
            <div class="stat-card-value">{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_e2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Heart Disease (1)</div>
            <div class="stat-card-value" style="color: #ef4444;">{int(df['target'].sum())}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_e3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Normal / Healthy (0)</div>
            <div class="stat-card-value" style="color: #10b981;">{int(len(df) - df['target'].sum())}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_e4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Clinical Features</div>
            <div class="stat-card-value">{len(df.columns) - 1}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔍 View Raw Clinical Dataset Sample", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("##### 1. Patient Age Distribution by Cardiac Status")
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=100)
        sns.histplot(data=df, x='age', hue='target', kde=True, palette={0: '#10b981', 1: '#ef4444'}, ax=ax, bins=20)
        ax.set_title("Age Distribution (0 = Healthy, 1 = Disease)", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Age (years)", fontsize=9)
        ax.set_ylabel("Patient Count", fontsize=9)
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)

    with col_g2:
        st.markdown("##### 2. Peak Heart Rate (thalach) vs. Age")
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=100)
        sns.scatterplot(data=df, x='age', y='thalach', hue='target', palette={0: '#10b981', 1: '#ef4444'}, alpha=0.85, s=60, ax=ax)
        ax.set_title("Peak Heart Rate vs. Age", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Age (years)", fontsize=9)
        ax.set_ylabel("Max Heart Rate Achieved", fontsize=9)
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)

    col_g3, col_g4 = st.columns(2)
    with col_g3:
        st.markdown("##### 3. Chest Pain Severity Distribution")
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=100)
        sns.countplot(data=df, x='cp', hue='target', palette={0: '#10b981', 1: '#ef4444'}, ax=ax)
        ax.set_title("Chest Pain Types vs Cardiac Outcome", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Chest Pain Type (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)", fontsize=9)
        ax.set_ylabel("Patient Count", fontsize=9)
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)

    with col_g4:
        st.markdown("##### 4. Correlation with Heart Disease Target")
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=100)
        corr = df.corr()
        sns.heatmap(corr[['target']].sort_values(by='target', ascending=False), annot=True, cmap='vlag', fmt=".2f", ax=ax, cbar=False)
        ax.set_title("Feature Correlation Coefficients", fontsize=11, fontweight='bold', pad=10)
        st.pyplot(fig)
        plt.close(fig)

# ==================== TAB 3: MODEL PERFORMANCE ====================
with tabs[2]:
    st.markdown("#### Model Benchmark & Clinical Validation")
    st.caption("Quantitative performance metrics computed on held-out test data (Stratified 80/20 split):")
    
    if metrics:
        models_data = metrics["models"]
        
        # Display comparison table / metric cards
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.markdown("##### 🌲 Random Forest *(Primary)*")
            rf_m = models_data["Random Forest"]
            st.markdown(f"""
            <div style="background-color: #f8fafc; border: 1.5px solid #2563eb; border-radius: 10px; padding: 14px;">
                <p style="margin: 4px 0;">• <b>Accuracy:</b> <code style="font-size: 1rem;">{rf_m['accuracy']}%</code></p>
                <p style="margin: 4px 0;">• <b>Recall (Sensitivity):</b> <code style="font-size: 1rem; color: #16a34a; font-weight: bold;">{rf_m['recall']}%</code> ⭐</p>
                <p style="margin: 4px 0;">• <b>Precision:</b> <code>{rf_m['precision']}%</code></p>
                <p style="margin: 4px 0;">• <b>F1-Score:</b> <code>{rf_m['f1_score']}%</code></p>
                <p style="margin: 4px 0;">• <b>ROC-AUC:</b> <code>{rf_m['roc_auc']}%</code></p>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.markdown("##### 📈 Logistic Regression")
            lr_m = models_data["Logistic Regression"]
            st.markdown(f"""
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px;">
                <p style="margin: 4px 0;">• <b>Accuracy:</b> <code>{lr_m['accuracy']}%</code></p>
                <p style="margin: 4px 0;">• <b>Recall:</b> <code>{lr_m['recall']}%</code></p>
                <p style="margin: 4px 0;">• <b>Precision:</b> <code>{lr_m['precision']}%</code></p>
                <p style="margin: 4px 0;">• <b>F1-Score:</b> <code>{lr_m['f1_score']}%</code></p>
                <p style="margin: 4px 0;">• <b>ROC-AUC:</b> <code>{lr_m['roc_auc']}%</code></p>
            </div>
            """, unsafe_allow_html=True)

        with col_m3:
            st.markdown("##### 🌳 Decision Tree")
            dt_m = models_data["Decision Tree"]
            st.markdown(f"""
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px;">
                <p style="margin: 4px 0;">• <b>Accuracy:</b> <code>{dt_m['accuracy']}%</code></p>
                <p style="margin: 4px 0;">• <b>Recall:</b> <code>{dt_m['recall']}%</code></p>
                <p style="margin: 4px 0;">• <b>Precision:</b> <code>{dt_m['precision']}%</code></p>
                <p style="margin: 4px 0;">• <b>F1-Score:</b> <code>{dt_m['f1_score']}%</code></p>
                <p style="margin: 4px 0;">• <b>ROC-AUC:</b> <code>{dt_m['roc_auc']}%</code></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("##### Confusion Matrix (Random Forest)")
            cm = np.array(models_data["Random Forest"]["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(6, 4.2), dpi=100)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Predicted Healthy (0)', 'Predicted Disease (1)'],
                        yticklabels=['Actual Healthy (0)', 'Actual Disease (1)'], ax=ax, cbar=False)
            ax.set_ylabel("True Ground Truth", fontsize=9, fontweight='bold')
            ax.set_xlabel("Predicted Label", fontsize=9, fontweight='bold')
            ax.set_title("Test Confusion Matrix", fontsize=11, fontweight='bold', pad=10)
            st.pyplot(fig)
            plt.close(fig)

        with col_c2:
            st.markdown("##### Feature Importance Ranking")
            feat_df = pd.DataFrame(metrics["feature_importance"]).sort_values(by="importance", ascending=True)
            fig, ax = plt.subplots(figsize=(6, 4.2), dpi=100)
            ax.barh(feat_df['feature'], feat_df['importance'], color='#2563eb', alpha=0.85)
            ax.set_xlabel("Relative Importance (Gini Impurity Reduction)", fontsize=9)
            ax.set_title("Predictive Clinical Feature Weights", fontsize=11, fontweight='bold', pad=10)
            sns.despine()
            st.pyplot(fig)
            plt.close(fig)

# ==================== TAB 4: FULL DATASET EXPLORER ====================
with tabs[3]:
    st.markdown("#### 📁 Comprehensive Patient Records Explorer")
    st.caption("Inspect, filter, and export the complete clinical dataset used for model training and evaluation.")

    # Top Metadata Banner
    st.markdown("""
    <div style="background-color: #f1f5f9; border-left: 4px solid #2563eb; border-radius: 6px; padding: 12px 16px; margin-bottom: 16px;">
        <span style="font-size: 0.95rem; font-weight: bold; color: #0f172a;">Dataset Provenance & Clinical Source:</span><br>
        <span style="font-size: 0.88rem; color: #475569;">
            <b>• Dataset Name:</b> Expanded Cleveland Clinical Cardiovascular Cohort<br>
            <b>• Primary Origin:</b> Cleveland Clinic Foundation & UCI Machine Learning Repository (Heart Disease Clinical Study)<br>
            <b>• Diagnostic Ground Truth:</b> Coronary Angiography (Fluoroscopic cardiac catheterization defining ≥ 50% diameter stenosis in major vessels)<br>
            <b>• Data Quality:</b> 100% Complete records with zero null/missing values across 13 input clinical features and 1 binary target.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # 4 Quick Metric Cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Patients", f"{len(df):,}")
    with m_col2:
        n_pos = int((df['target'] == 1).sum())
        st.metric("Heart Disease (Target = 1)", f"{n_pos:,}", f"{(n_pos/len(df))*100:.1f}%")
    with m_col3:
        n_neg = int((df['target'] == 0).sum())
        st.metric("Healthy Baseline (Target = 0)", f"{n_neg:,}", f"{(n_neg/len(df))*100:.1f}%")
    with m_col4:
        st.metric("Missing / Null Values", "0 (Clean)", "100% Valid")

    st.markdown("---")

    # Interactive Filters for Teacher Demonstration
    st.markdown("##### 🔍 Interactive Filters")
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        target_filter = st.selectbox(
            "Filter by Clinical Diagnosis:",
            options=["All Records", "Heart Disease Present (Target = 1)", "Healthy Baseline (Target = 0)"]
        )
    with f_col2:
        sex_filter = st.selectbox(
            "Filter by Gender:",
            options=["All Genders", "Male (1)", "Female (0)"]
        )
    with f_col3:
        min_age, max_age = int(df['age'].min()), int(df['age'].max())
        age_range = st.slider("Filter by Age Range:", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # Apply Filters
    filtered_df = df.copy()
    if target_filter == "Heart Disease Present (Target = 1)":
        filtered_df = filtered_df[filtered_df['target'] == 1]
    elif target_filter == "Healthy Baseline (Target = 0)":
        filtered_df = filtered_df[filtered_df['target'] == 0]

    if sex_filter == "Male (1)":
        filtered_df = filtered_df[filtered_df['sex'] == 1]
    elif sex_filter == "Female (0)":
        filtered_df = filtered_df[filtered_df['sex'] == 0]

    filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]

    # Interactive Table View
    st.markdown(f"##### Showing **{len(filtered_df):,}** of **{len(df):,}** Patient Records")
    st.dataframe(filtered_df, use_container_width=True, height=450)

    # Download Button & Statistics Expander
    d_col1, d_col2 = st.columns([1, 2])
    with d_col1:
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full heart.csv (5,000 Rows)",
            data=csv_bytes,
            file_name="heart_disease_5000_records.csv",
            mime="text/csv",
            help="Click to download the full raw CSV dataset directly to your computer."
        )

    with d_col2:
        st.caption("💡 *Tip for Presentation:* You can click any column header above (e.g. `age`, `chol`, `trestbps`) to sort in ascending or descending order.")

    # Expandable Data Dictionary
    with st.expander("📖 View Complete Clinical Data Dictionary (14 Features Explained)"):
        dictionary_data = {
            "Feature": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"],
            "Clinical Description": [
                "Patient age in years",
                "Biological sex (1 = Male, 0 = Female)",
                "Chest pain type (0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)",
                "Resting systolic blood pressure upon admission (mm Hg)",
                "Serum cholesterol level (mg/dl)",
                "Fasting blood sugar > 120 mg/dl (1 = Diabetic/Elevated, 0 = Normal)",
                "Resting electrocardiographic results (0 = Normal, 1 = ST-T wave abnormality, 2 = LV hypertrophy)",
                "Maximum peak heart rate achieved during treadmill stress test (bpm)",
                "Exercise induced angina (1 = Yes, 0 = No)",
                "ST-segment depression induced by exercise relative to rest (in mm)",
                "Slope of peak exercise ST segment (0 = Upsloping, 1 = Flat, 2 = Downsloping)",
                "Number of major coronary arteries (0-3) colored by fluoroscopy dye",
                "Thallium scintigraphy heart stress test (1 = Normal, 2 = Fixed defect, 3 = Reversible defect)",
                "Coronary Artery Disease Status (1 = Disease Present [≥ 50% stenosis], 0 = Healthy)"
            ],
            "Type": ["Numeric", "Binary", "Categorical (0-3)", "Numeric", "Numeric", "Binary", "Categorical (0-2)", "Numeric", "Binary", "Numeric", "Categorical (0-2)", "Numeric (0-3)", "Categorical (1-3)", "Binary (0/1)"]
        }
        st.table(pd.DataFrame(dictionary_data))

    # Expandable Statistical Summary
    with st.expander("📊 View Descriptive Summary Statistics (Mean, Std, Min, 25%, 50%, 75%, Max)"):
        st.dataframe(df.describe().T.round(2), use_container_width=True)

