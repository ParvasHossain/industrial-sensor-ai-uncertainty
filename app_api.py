from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from data_and_ml import rf_model, scaler

app = FastAPI(title="Sensor Uncertainty & Failure Prediction API")

class SensorData(BaseModel):
    temperature: float
    vibration: float
    pressure: float

@app.get("/")
def home():
    return {"status": "Active", "message": "Sensor AI API is Running"}

@app.post("/predict")
def predict_failure(data: SensorData):
    input_data = np.array([[data.temperature, data.vibration, data.pressure]])
    scaled_data = scaler.transform(input_data)
    
    prediction = rf_model.predict(scaled_data)[0]
    probability = rf_model.predict_proba(scaled_data)[0][1]
    
    return {
        "failure_predicted": int(prediction),
        "failure_probability": round(float(probability), 4),
        "status": "Warning: Maintenance Required!" if prediction == 1 else "Normal Operations"
    }