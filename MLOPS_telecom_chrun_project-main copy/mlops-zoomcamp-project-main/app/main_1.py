from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("models/model.pkl")

class PredictionInput(BaseModel):
    # Define your input features here

@app.post("/predict")
async def predict(input: PredictionInput):
    # Transform input to model features
    features = [input.feature1, input.feature2, ...]
    prediction = model.predict([features])[0]
    return {"prediction": prediction}