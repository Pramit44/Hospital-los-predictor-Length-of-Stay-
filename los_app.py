import streamlit as st
import pandas as pd
import joblib
import json

model = joblib.load('xgboost_los_model.pkl')
with open('model_columns.json', 'r') as f:
    model_columns = json.load(f)

# Page config
st.set_page_config(page_title="Hospital LOS Predictor", layout="wide")
st.title("🏥 Hospital Length of Stay (LOS) Predictor")
st.markdown("Predict the number of days a patient will stay in the hospital using our optimized Machine Learning model.")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("👤 Basic Details & Facility")
    gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1])
    facid = st.selectbox("Facility ID", ["A", "B", "C", "D", "E"])

with col2:
    st.subheader("❤️ Vitals")
    respiration = st.selectbox("Respiration", ["Low", "Normal", "High"])
    # Added "Low_Pulse" as a base category
    pulse = st.selectbox("Pulse", ["Low_Pulse", "Normal_Pulse", "Tachycardia"])
    # Added "Underweight" as a base category
    bmi = st.selectbox("Body Mass Index", ["Underweight", "Normal", "Overweight", "Obesity"])

st.markdown("---")

col3, col4 = st.columns(2)
with col3:
    st.subheader("🧪 Lab Results")
    hematocrit = st.selectbox("Hematocrit Levels", ["Moderate Anemia", "Mild Anemia", "Normal", "High (Polycythemia)"])
    neutrophils = st.selectbox("White Blood Cells (Neutrophils)", ["Low", "Normal", "High"])
    sodium = st.selectbox("Sodium Level", ["Low", "Normal", "High"])
    # Added "Normal" to Glucose
    glucose = st.selectbox("Glucose Levels", ["Normal", "Prediabetes", "Diabetes"])
    bloodureanitro = st.selectbox("Blood Urea Nitrogen", ["Low", "Normal", "High"])
    creatinine = st.selectbox("Creatinine", ["Low", "Normal", "High"])

with col4:
    st.subheader("⚠️ Risk Profile & Diagnosis")
    rcount_display = st.selectbox("Readmission Count", ["0", "1", "2", "3", "4+"])
    # Added "None / Low-Load" to Secondary Diagnosis
    secondary_diag = st.selectbox("Secondary Diagnosis", ["None / Low-Load", "Moderate-Load Routine Case", "Critical High-Load Condition"])

st.markdown("---")

st.subheader("📋 Medical Indicators & History")
st.write("Select all that apply to the patient:")

cond_col1, cond_col2, cond_col3 = st.columns(3)
with cond_col1:
    dialysis = st.checkbox("Dialysis Indicator")
    asthma = st.checkbox("Asthma Indicator")
    irondef = st.checkbox("Iron Deficiency")
    pneum = st.checkbox("Pneumonia")

with cond_col2:
    substance = st.checkbox("Substance Dependency")
    psych_major = st.checkbox("Psychological Disorder")
    depress = st.checkbox("Depression")
    psychother = st.checkbox("Other Psychological Disorders")

with cond_col3:
    fibrosis = st.checkbox("Fibrosis")
    malnutrition = st.checkbox("Malnutrition")
    hemo = st.checkbox("Hemo")

st.markdown("---")
if st.button("Predict Stay Duration", type="primary", use_container_width=True):

    rcount_mapping = {
        "0": "Low Risk",
        "1": "Moderate Risk",
        "2": "High Risk",
        "3": "High Super-Utilizer",
        "4+": "Extreme Super-Utilizer"
    }
    rcount = rcount_mapping[rcount_display]

    # Initialize all columns with 0
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0

    # Map binary checkboxes and direct values
    input_data['gender'] = gender
    input_data['dialysisrenalendstage'] = int(dialysis)
    input_data['asthma'] = int(asthma)
    input_data['irondef'] = int(irondef)
    input_data['pneum'] = int(pneum)
    input_data['substancedependence'] = int(substance)
    input_data['psychologicaldisordermajor'] = int(psych_major)
    input_data['depress'] = int(depress)
    input_data['psychother'] = int(psychother)
    input_data['fibrosisandother'] = int(fibrosis)
    input_data['malnutrition'] = int(malnutrition)
    input_data['hemo'] = int(hemo)

    # Dynamic mapping: Agar option JSON mein nahi hai (like 'Normal' glucose), toh wo ignore ho jayega (yaani 0 rahega)
    dummy_mappings = [
        f"facid_{facid}",
        f"respiration_{respiration}",
        f"bmi_{bmi}",
        f"hematocrit_{hematocrit}",
        f"neutrophils_{neutrophils}",
        f"sodium_{sodium}",
        f"glucose_{glucose}",
        f"bloodureanitro_{bloodureanitro}",
        f"creatinine_{creatinine}",
        f"secondarydiagnosisnonicd9_{secondary_diag}",
        f"rcount_{rcount}",
        f"pulse_{pulse}"
    ]

    for col in dummy_mappings:
        if col in input_data.columns:
            input_data[col] = 1

    input_data = input_data[model_columns]

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"### 🏨 Predicted Length of Stay: {prediction:.1f} Days")
    except Exception as e:
        st.error(f"Error making prediction: {e}")





















