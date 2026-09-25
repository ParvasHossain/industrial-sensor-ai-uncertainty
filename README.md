# 🛠️ Industrial Sensor AI & Uncertainty Estimation System

An end-to-end Machine Learning & Deep Learning solution for industrial sensor predictive maintenance, anomaly alerting, and measurement uncertainty estimation.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---

## 📌 Project Overview

In critical industrial environments (e.g., turbines, heavy pumps, calibration labs), sudden machine breakdown leads to costly downtime and safety hazards. This project demonstrates an **AI-driven Predictive Maintenance System** combined with **Statistical Measurement Uncertainty Analysis**.

The system evaluates real-time multi-sensor telemetry (Temperature, Vibration, Pressure), computes measurement error metrics (Standard Error of the Mean, Interquartile Range), and classifies operational risk using both Classical ML (Random Forest) and Deep Learning (PyTorch Neural Networks).

---

## ✨ Key Features

- **Measurement Uncertainty Analysis**: Leverages `SciPy` and `NumPy` to compute statistical metrics, including Standard Deviation ($s$), Standard Error of Mean ($\text{SEM}$), and Interquartile Range ($\text{IQR}$) for noise and drift detection.
- **Dual AI/ML Inference Engine**:
  - **Random Forest Classifier**: Handles non-linear feature interaction with high accuracy.
  - **PyTorch Neural Network**: Implements a Multi-Layer Perceptron (MLP) for continuous probability scoring.
- **RESTful API Backend**: Powered by `FastAPI` to expose asynchronous inference endpoints for live sensor streaming.
- **Interactive Web Dashboard**: Built using `Streamlit` to simulate live sensor adjustments, plot scatter distributions via `Seaborn`/`Matplotlib`, and trigger real-time failure alerts.

---

## 🏗️ System Architecture

```text
       ┌────────────────────────┐
       │ Sensor Telemetry Stream│
       └───────────┬────────────┘
                   │
  ┌────────────────┴────────────────┐
  │  Uncertainty Budgeting & Math   │
  │     (NumPy / SciPy / Pandas)    │
  └────────────────┬────────────────┘
                   │
  ┌────────────────┴────────────────┐
  │      AI Model Inference         │
  │  (Scikit-Learn & PyTorch DL)    │
  └────────┬───────────────┬────────┘
           │               │
┌──────────▼────────┐   ┌──▼─────────────────────────┐
│ FastAPI Service   │   │ Streamlit Monitoring UI    │
│ (Production API)  │   │ (Interactive Dashboard)     │
└───────────────────┘   └────────────────────────────┘
```

---

## 🛠️ Tech Stack & Libraries

- **Data Processing & Mathematics**: `NumPy`, `Pandas`, `SciPy`
- **Data Visualization**: `Matplotlib`, `Seaborn`
- **Machine Learning & AI**: `Scikit-Learn`, `PyTorch`
- **Deployment & Interface**: `FastAPI`, `Uvicorn`, `Streamlit`

---

## ⚙️ Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/industrial-sensor-ai-uncertainty.git
cd industrial-sensor-ai-uncertainty
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Steps

### Step 1: Train Models & Generate Assets
Execute the core data pipeline to process sensor readings, calculate uncertainty metrics, and train both Random Forest and PyTorch models:
```bash
python data_and_ml.py
```

### Step 2: Start the FastAPI Backend
Launch the API server for model serving:
```bash
uvicorn app_api:app --reload
```
*Interactive Swagger documentation available at `http://127.0.0.1:8000/docs`.*

### Step 3: Launch the Streamlit Monitoring Dashboard
Open a new terminal window, activate `venv`, and run:
```bash
streamlit run app_dashboard.py
```

---

## 📊 Dashboard Preview

The dashboard displays:
1. **Statistical Uncertainty Metrics** in the left sidebar.
2. **Sensor Data Distribution Plots** highlighting normal vs. anomalous failure clusters.
3. **Live Prediction Controls** to simulate parameter variations ($T$, $V$, $P$) and receive immediate maintenance warnings.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.