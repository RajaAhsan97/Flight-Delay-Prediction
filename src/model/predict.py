import pandas as pd
import os
import joblib
import numpy as np


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


def predict(airline, origin, dest, distance, dep_delay,
            taxi_out, taxi_in, day_of_week, day_of_month,
            carrier_delay=0, weather_delay=0):
    
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

    if scaler: # if logistic regression model
        # scaled input
        X_inp = scaler.transform(X_inp)
    

    probability  = model.predict_proba(X_inp)[0][1]
    prediction  = model.predict(X_inp)[0]

    prediction = "DELAYED" if prediction==1 else "ON TIME"

    return prediction, float(probability), 


if __name__ == "__main__":
    pred, prob = predict(airline='AA', origin='ATL', dest='LAX',
                        distance=1946, dep_delay=20,
                        taxi_out=18, taxi_in=9,
                        day_of_week=5, day_of_month=15
                )

    print(f"\n Predicted Arival: {pred}")
    print(f"\n Predicted Probability: {prob}")

# run this code
# python -m src.model.predict