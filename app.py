import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration & Theme Optimization
st.set_page_config(
    page_title="Snakebite CDSS Pro",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical Theme Styling via CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; height: 3rem; }
    .reportview-container .main .block-container{ padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 2. Deserialization Layer (Load Model & Encoders)
@st.cache_resource
def load_artifacts():
    return joblib.load('snakebite_triage_model.pkl')

try:
    artifacts = load_artifacts()
    model = artifacts['model']
    gender_encoder = artifacts['gender_encoder']
    venom_encoder = artifacts['venom_encoder']
except FileNotFoundError:
    st.error("❌ 'snakebite_triage_model.pkl' missing. Please run your training model script first.")
    st.stop()

# 3. Sidebar Panel for Metadata & System Logs
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-doctor.png", width=80)
    st.markdown("## **Clinical Station Portal**")
    st.info("📍 **Location:** Rural Emergency Triage Hub")
    st.markdown("---")
    st.markdown("### **System Guidelines**")
    st.caption("This Clinical Decision Support System (CDSS) aligns with the National Health Mission (NHM) snakebite management guidelines.")
    st.markdown("---")
    st.caption("🔒 Secured EMR Data Pipeline Instance")

# 4. Main UI Dashboard Layout
st.title("⚕️ Snakebite Clinical Decision Support System (CDSS)")
st.markdown("### **Real-Time Patient Triage & Autonomous Treatment Protocol**")
st.markdown("---")

# Organized Grid for Input Parameters
st.markdown("#### **🩺 Patient Admission & Vitals Intake**")
row1_col1, row1_col2, row1_col3 = st.columns(3)
row2_col1, row2_col2, row2_col3 = st.columns(3)

with row1_col1:
    age = st.number_input("Patient Age (Years)", min_value=1, max_value=100, value=28, step=1)
with row1_col2:
    gender = st.selectbox("Biological Gender", ["Male", "Female"])
with row1_col3:
    time_delay = st.number_input("Time Elapsed Since Bite (Mins)", min_value=0, max_value=720, value=45, step=5)

with row2_col1:
    heart_rate = st.number_input("Heart Rate (BPM)", min_value=40, max_value=200, value=78, step=1)
with row2_col2:
    systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=50, max_value=200, value=122, step=1)
with row2_col3:
    local_swelling = st.selectbox("Local Swelling Severity", ["None", "Mild", "Severe"])

neuro_signs = st.radio(
    "**Neurological Evaluation Checklist (Ptosis / Bulbar Palsy / Muscle Weakness):**",
    ["Absent", "Present"],
    horizontal=True
)

# 5. Data Vector Preprocessing 
neuro_encoded = 1 if neuro_signs == "Present" else 0
gender_encoded = gender_encoder.transform([gender])[0]
swelling_map = {'None': 0, 'Mild': 1, 'Severe': 2}
swelling_encoded = swelling_map[local_swelling]

input_data = pd.DataFrame([{
    'Age': age,
    'Gender': gender_encoded,
    'Time_Since_Bite_Mins': time_delay,
    'Heart_Rate_BPM': heart_rate,
    'Systolic_BP': systolic_bp,
    'Local_Swelling': swelling_encoded,
    'Neurological_Signs': neuro_encoded
}])

st.markdown("---")

# 6. Prediction Engine & Dynamic Reporting Layout
if st.button("🚀 Process Vitals & Generate Diagnostic Protocol", type="primary"):
    
    # Run predictions across all 3 targets simultaneously
    prediction = model.predict(input_data)[0]
    pred_severity = prediction[0]
    pred_venom_encoded = prediction[1]
    pred_vials = prediction[2]
    
    pred_venom_str = venom_encoder.inverse_transform([pred_venom_encoded])[0]

    if pred_severity == 0 or pd.isna(pred_venom_str) or pred_venom_str == 'None':
        pred_venom_str = "No Venom Detected"
    
    st.markdown("### **🏥 Diagnostic Evaluation & Action Matrix**")
    
    # Visual KPI Cards for Quick Assessment Summary
    m_col1, m_col2, m_col3 = st.columns(3)
    
    # Setup styles based on risk categories
    if pred_severity == 0:
        status_label, color_theme, banner = "Low Risk / Dry Bite", "normal", st.success
    elif pred_severity == 1:
        status_label, color_theme, banner = "Moderate Risk", "off", st.warning
    else:
        status_label, color_theme, banner = "Critical Shock Risk", "inverse", st.error
        
    with m_col1:
        st.metric(label="📊 Assigned Triage Tier", value=status_label)
    with m_col2:
        st.metric(label="🧪 Identified Venom Target", value=pred_venom_str)
    with m_col3:
        st.metric(label="🧪 Polyvalent ASV Required", value=f"{pred_vials} Vials")
        
    st.markdown("---")
    
    # Detailed Treatment Execution Guidelines
    banner("#### **📋 Clinical Action Directive Sheet**")
    
    if pred_severity == 0:
        st.markdown(f"**Pathology Evaluation:** Defensive or dry bite suspected. No systemic envenomation patterns detected.")
        st.info("📌 **Mandatory Protocol:** **Do NOT administer Anti-Snake Venom (ASV).** Admit the patient for a mandatory **24-hour ward observation window**. Re-evaluate clinical parameters and check blood coagulation profiles every 30 minutes to safeguard against delayed symptom onset.")
        
    elif pred_severity == 1:
        st.markdown(f"**Pathology Evaluation:** Neurotoxic envenomation detected. (Indicative of Cobra or Krait neurotoxin targeting neuromuscular junctions).")
        st.markdown("⚠️ **Emergency Operational Steps:**")
        st.markdown(f"1. **Initiate ASV Therapy:** Secure IV access immediately and administer **{pred_vials} Vials of Polyvalent ASV** via slow infusion.")
        st.markdown("2. **Neuromuscular Protection:** Keep an **Atropine-Neostigmine (AN)** regimen calculated and on standby to counter escalating paralysis loops.")
        st.markdown("3. **Respiratory Safeguard:** Monitor oxygen saturation levels closely. Ensure a bag-valve-mask and mechanical ventilation equipment are moved immediately to the bedside.")
        
    elif pred_severity == 2:
        st.markdown(f"**Pathology Evaluation:** Hemotoxic envenomation with progressive cardiovascular collapse signs. (Indicative of Russell's or Saw-scaled Viper toxins damaging vascular linings).")
        st.markdown("🚨 **Immediate Life-Support Measures Required:**")
        st.markdown(f"1. **High-Dose ASV Loading:** Administer **{pred_vials} Vials of Polyvalent ASV** via IV infusion to handle systemic venom burden.")
        st.markdown(f"2. **Resuscitation:** Patient shows vital markers of shock (Heart Rate: {heart_rate} BPM | Systolic BP: {systolic_bp} mmHg). Initiate immediate, rapid IV crystalloid fluid resuscitation.")
        st.markdown("3. **Coagulopathy Management:** Execute a **20-Minute Whole Blood Clotting Test (20WBCT)** at the bedside immediately. Screen continuously for hemorrhage signs (gums, hematuria) and protect against Acute Kidney Injury (AKI) by monitoring hourly urine output metrics.")