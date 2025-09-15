from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="ML Model API (Starter)")

MODEL_PATH = os.getenv('MODEL_PATH', 'model.pkl')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train.py to create it.")

model = joblib.load(MODEL_PATH)

class InputData(BaseModel):
    features: list

@app.get('/')
def read_root():
    return {"message": "API is running"}

# @app.get('/health')
# def health():
    # return {'status': 'ok'}


@app.get("/health")
def health_check():
    return {"status": "Service is running ✅"}

@app.post('/predict')
def predict(data: InputData):
    if not isinstance(data.features, list):
        return {"error": "Features should be a list of numbers."}
    
    x = [data.features]
    pred = model.predict(x)
    return {'prediction': float(pred[0])}
