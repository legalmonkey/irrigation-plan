from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os

app = FastAPI()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment variables")

genai.configure(api_key=API_KEY)

class SensorData(BaseModel):
    temperature: float
    humidity: float
    soil: float
    crop: str
    location: str
    farm_size: float
    translate: bool

@app.get("/")
def root():
    return {"status": "Backend running"}

def generate_advice(data: SensorData):

    prompt = f"""
    You are a Real-Time Irrigation AI for a farmer in {data.location}.
    Crop: {data.crop} | Farm Size: {data.farm_size} Acres.

    LATEST SENSOR READING:
    Temp: {data.temperature}
    Humidity: {data.humidity}
    Soil: {data.soil}

    TASK:
    1. Analyze this specific hour's data.
    2. If soil is dry, give EXACT liters needed for {data.farm_size} acres.
    3. Formula: 1mm = 4046 Liters per Acre.

    STRICT FORMAT:
    -------------------------------------------
    CURRENT STATUS:
    - Readings: {data.temperature}, {data.humidity}, {data.soil}
    - Status: [brief explanation]

    IMMEDIATE ACTION:
    1. WATERING: [Apply Xm now / No water needed]
    2. TOTAL VOLUME: [X] Liters for {data.farm_size} acres.
    3. NEXT CHECK: [Advice for next hour]
    -------------------------------------------

    {"Translate the entire output to the local language." if data.translate else "Use simple English."}
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Processing Error: {str(e)}"

@app.post("/predict")
def predict(data: SensorData):
    advice = generate_advice(data)
    return {"advice": advice}