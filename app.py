from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import Dict, Any
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(
    title="Accident Fatality Prediction API",
    description="An API to predict the severity of traffic accidents based on various factors using a Random Forest model."
)

app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "artifacts/rf_model.pkl"
ENCODERS_PATH = "artifacts/label_encoders.pkl"
COLUMNS_PATH = "artifacts/feature_columns.pkl"

# Global references
model = None
encoders = None
feature_columns = None
target_encoder = None

@app.on_event("startup")
def load_artifacts():
    global model, encoders, feature_columns, target_encoder
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH) or not os.path.exists(COLUMNS_PATH):
        print("⚠️ Warning: Model artifacts not found. Please run 'python train_model.py' first.")
        return
    
    print("✅ Loading model and artifacts into memory...")
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
    
    target_encoder = encoders.get('Accident_Severity')
    print("✅ System Ready!")

class PredictionInput(BaseModel):
    # A generic dictionary since there are 20+ columns
    features: Dict[str, Any]

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/features")
def get_features():
    features_info = []
    if feature_columns:
        for col in feature_columns:
            if encoders and col in encoders:
                # Carefully extract options, converting them to native python strings to prevent JSON float NaN errors
                raw_options = encoders[col].classes_.tolist()
                options = []
                for opt in raw_options:
                    # Ignore np.nan or float nan values which break fastapi json engine
                    if pd.isna(opt):
                        continue
                    options.append(str(opt))
                features_info.append({"name": col, "type": "categorical", "options": options})
            else:
                features_info.append({"name": col, "type": "numeric"})
    return {"features": features_info}

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Train the model using train_model.py first.")
    
    data_dict = input_data.features
    
    # Gracefully fill in any omitted features with default 0 value
    for col in feature_columns:
        if col not in data_dict:
            data_dict[col] = '0' 
            
    # Load into DataFrame (as model expects DataFrame for named features)
    df = pd.DataFrame([data_dict])
    df = df[feature_columns]
    
    # Dynamically encode features the same way it was trained
    for col in feature_columns:
        if col in encoders:
            label_encoder = encoders[col]
            df[col] = df[col].astype(str)
            
            # Edge case handling: If web user submits a class that the model has never seen before, 
            # we default it to the First known class (usually 0/nan class) to prevent 500 crashes.
            known_classes = set(label_encoder.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else label_encoder.classes_[0])
            
            df[col] = label_encoder.transform(df[col])
            
    # Execute Model Inference
    prediction = model.predict(df)
    
    # Decode prediction back into human readable status (e.g. "Serious", "Slight")
    result_class = int(prediction[0])
    if target_encoder:
        prediction_label = target_encoder.inverse_transform([result_class])[0]
    else:
        prediction_label = str(result_class)
        
    return {
        "status": "success",
        "predicted_severity": prediction_label
    }
