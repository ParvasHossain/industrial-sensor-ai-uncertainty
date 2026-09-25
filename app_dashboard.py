import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_and_ml import df, temp_std, temp_sem, rf_model, scaler

st.set_page_config(page_title="Sensor Uncertainty & AI Dashboard", layout="wide")

st.title("🛠️ Industrial Sensor AI & Uncertainty Dashboard")
st.write("This dashboard leverages NumPy, Pandas, SciPy, Scikit-Learn, PyTorch & Streamlit.")

# Sidebar Metrics & Uncertainty
st.sidebar.header("Statistical Metrics (SciPy/NumPy)")
st.sidebar.metric("Temperature Std Dev", f"{temp_std:.2f}")
st.sidebar.metric("Temperature SEM (Uncertainty)", f"{temp_sem:.4f}")

# Main Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sensor Data Sample")
    st.dataframe(df.head(10))

with col2:
    st.subheader("Data Distribution Plot")
    st.image('sensor_distribution.png')

# Real-time Interactive Prediction
st.markdown("---")
st.subheader("🔮 Live Sensor Prediction System")

c1, c2, c3 = st.columns(3)
input_temp = c1.number_input("Temperature (°C)", value=75.0)
input_vib = c2.number_input("Vibration (g)", value=0.6)
input_pres = c3.number_input("Pressure (kPa)", value=98.0)

if st.button("Predict Maintenance Status"):
    sample = np.array([[input_temp, input_vib, input_pres]])
    sample_scaled = scaler.transform(sample)
    
    pred = rf_model.predict(sample_scaled)[0]
    prob = rf_model.predict_proba(sample_scaled)[0][1]
    
    if pred == 1:
        st.error(f"🚨 Warning: High Risk of Failure! Probability: {prob*100:.1f}%")
    else:
        st.success(f"✅ System Healthy. Failure Probability: {prob*100:.1f}%")