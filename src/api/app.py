# # run this
# # uvicorn src.api.app:app --reload

# src/api/app.py

import os
import time
import logging
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ===== FastAPI setup =====
app = FastAPI(title="Flight Delay Prediction API")

# ===== Load models at atartup =====
def load_model():
    """
    Load trained models from models folder
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # label encoder path
    le_path = os.path.join(
        script_dir,
        "..", "..",
        "models", 
        "label_encoders.pkl"
    )
    le_path = os.path.abspath(le_path)
    le = joblib.load(le_path)

    # model path
    model_path = os.path.join(
        script_dir,
        "..", "..",
        "models",
        "XGBoost_clf.pkl"
    )
    model_path = os.path.abspath(model_path)
    model_metadata = joblib.load(model_path)
    model = model_metadata['model']
    model_name = model_metadata['model_name']

    if model_name == 'LogisticRegression':
        scaler_path = os.path.join(
            script_dir,
            "..", "..",
            "models",
            "best_clf_scaler.pkl"
        )
        scaler_path = os.path.abspath(scaler_path)

        scaler = joblib.load(scaler_path)
        return model, le, scaler
    else:
        return model, le, None


# ================================================================
# Request Schema  (Input)
# ================================================================
class FlightInput(BaseModel):
    airline     : str   
    origin      : str   
    dest        : str   
    distance    : float 
    dep_delay   : float 
    taxi_out    : float 
    taxi_in     : float 
    day_of_week : int   
    day_of_month: int   
    carrier_delay: float 
    weather_delay: float


# ===== Health check endpoint =====
@app.get("/")
def read_root():
    return {"message": "Flight Delay Prediction API is running!"}

# ===== API endpoint =====
@app.post("/predict-delay")
def predict(request: FlightInput):
    airline = request.airline
    origin = request.origin 
    dest = request.dest
    distance = request.distance 
    dep_delay = request.dep_delay
    taxi_out = request.taxi_out
    taxi_in = request.taxi_in
    day_of_week = request.day_of_week
    day_of_month = request.day_of_month
    carrier_delay = request.carrier_delay
    weather_delay = request.weather_delay


    model, label_encoders, scaler = load_model()

    is_weekend   = 1 if day_of_week >= 6 else 0
    route        = f'{origin}_{dest}'
    dist_bucket  = pd.cut([distance],
                          bins=[0,500,1000,1500,2000,10000],
                          labels=['VeryShort','Short','Medium','Long','VeryLong'])[0]

    def safe_encode(enc, val):
        classes = list(enc.classes_)
        return enc.transform([val])[0] if val in classes else -1

    sample = {
        'OP_UNIQUE_CARRIER_ENC' : safe_encode(label_encoders['OP_UNIQUE_CARRIER'], airline),
        'ORIGIN_ENC'            : safe_encode(label_encoders['ORIGIN'], origin),
        'DEST_ENC'              : safe_encode(label_encoders['DEST'], dest),
        'ROUTE_ENC'             : safe_encode(label_encoders['ROUTE'], route),
        'DISTANCE'              : distance,
        'DIST_BUCKET_ENC'       : safe_encode(label_encoders['DIST_BUCKET'], str(dist_bucket)),
        'TAXI_OUT'              : taxi_out,
        'TAXI_IN'               : taxi_in,
        'DAY_OF_WEEK'           : day_of_week,
        'DAY_OF_MONTH'          : day_of_month,
        'IS_WEEKEND'            : is_weekend,
        'CARRIER_DELAY'         : carrier_delay,
        'WEATHER_DELAY'         : weather_delay,
        'DEP_DELAY'             : dep_delay,
    }
    X_inp = pd.DataFrame([sample])


    prediction = model.predict(X_inp)
    label = "DELAYED" if prediction == 1 else "ON TIME"

    probability  = model.predict_proba(X_inp)[0][1]
    if probability >= 0.75 or probability <= 0.25:
        confidence = "High"
    elif probability >= 0.60 or probability <= 0.40:
        confidence = "Medium"
    else:
        confidence = "Low"


    return {"Label": label, "confidence": confidence}


# run this
# uvicorn src.api.app:app --reload