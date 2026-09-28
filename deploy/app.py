import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostRegressor

app = FastAPI(title="Tourism Prediction API", version="1.0")

# Load models and artifacts at startup
MODELS_DIR = os.getenv("MODELS_DIR", "saved_models")

pass1_model = CatBoostRegressor()
pass1_model.load_model(os.path.join(MODELS_DIR, "catboost_pass1.cbm"))

final_model = CatBoostRegressor()
final_model.load_model(os.path.join(MODELS_DIR, "catboost_final.cbm"))

with open(os.path.join(MODELS_DIR, "pref_month_mean_reg.pkl"), "rb") as f:
    pref_month_mean_reg = pickle.load(f)

with open(os.path.join(MODELS_DIR, "pref_month_std_reg.pkl"), "rb") as f:
    pref_month_std_reg = pickle.load(f)

with open(os.path.join(MODELS_DIR, "smear_factor_cb.pkl"), "rb") as f:
    smear_factor_cb = pickle.load(f)

with open(os.path.join(MODELS_DIR, "pass1_feature_cols.pkl"), "rb") as f:
    features_pass1 = pickle.load(f)

with open(os.path.join(MODELS_DIR, "reg_feature_cols.pkl"), "rb") as f:
    features_final = pickle.load(f)

class PredictionRequest(BaseModel):
    prefecture: str
    month: int
    # Features
    month_sin: float
    month_cos: float
    year_trend: float
    temp_avg: float
    precip_sum: float
    cherry_blossom: int
    golden_week: int
    obon: int
    covid_shock: int
    lag_1: float
    lag_2: float
    lag_6: float
    lag_12: float
    roll_3_mean: float
    hub_lag_1: float
    lat: float
    lon: float
    coastal: int
    snow_region: int
    intl_airport: int
    dist_tokyo: float
    dist_osaka: float
    dist_osaka: float

@app.get("/")
def health_check():
    return {"status": "ok", "message": "CatBoost API is running"}

@app.post("/predict")
def predict(data: PredictionRequest):
    # Convert input to DataFrame (1 row)
    req_dict = data.model_dump()
    
    # Extract meta fields not used as features directly
    pref = req_dict.pop("prefecture")
    month = req_dict.pop("month")
    
    # Compute derived features dynamically
    pref_month_enc = pref_month_mean_reg.get((pref, month), 0.0)
    prev_m_for_lag1 = month - 1 if month > 1 else 12
    prev_m_enc = pref_month_mean_reg.get((pref, prev_m_for_lag1), 0.0)
    
    req_dict["pref_month_enc"] = pref_month_enc
    req_dict["lag_12_detrend"] = req_dict["lag_12"] - pref_month_enc
    req_dict["lag_1_detrend"] = req_dict["lag_1"] - prev_m_enc
    
    # We will compute seasonal_deviation via pass 1
    # Features exactly match the loaded columns for pass 1
    row_df = pd.DataFrame([req_dict])
    
    # Ensure correct column order for pass 1
    X_pass1 = row_df[features_pass1]
    
    # 1. Predict pass 1
    pred_pass1_log = pass1_model.predict(X_pass1)[0]
    
    # 2. Compute seasonal_deviation
    std = pref_month_std_reg.get((pref, month), 1.0)
    if std == 0:
        std = 1e-6
    
    corrected_seasonal_deviation = (pred_pass1_log - pref_month_enc) / std
    
    # 3. Predict final pass
    row_df["seasonal_deviation"] = corrected_seasonal_deviation
    
    # Use features_pass1 + seasonal_deviation (26 columns) 
    # instead of features_final which only had 22 columns saved in the pickle
    features_final_corrected = features_pass1 + ["seasonal_deviation"]
    X_pass2 = row_df[features_final_corrected]
    
    pred_final_log = final_model.predict(X_pass2)[0]
    
    # 4. Apply Duan's Smearing Estimator
    pred_actual = max(smear_factor_cb * np.expm1(pred_final_log), 0)
    
    return {
        "prefecture": pref,
        "month": month,
        "log_prediction_pass1": float(pred_pass1_log),
        "seasonal_deviation": float(corrected_seasonal_deviation),
        "log_prediction_final": float(pred_final_log),
        "predicted_stays": int(round(pred_actual))
    }
