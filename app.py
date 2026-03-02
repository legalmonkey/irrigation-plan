from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os

app = FastAPI()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

genai.configure(api_key=API_KEY)

# ---- STORE LATEST DATA ----
latest_data = {}

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

@app.get("/latest")
def get_latest():
    return latest_data

def generate_advice(data: SensorData):

    prompt = f"""
    You are a Real-Time Irrigation AI for a farmer in {data.location}.
    Crop: {data.crop} | Farm Size: {data.farm_size} Acres.

    LATEST SENSOR READING:
    Temp: {data.temperature}
    Humidity: {data.humidity}
    Soil: {data.soil}

    1mm = 4046 Liters per Acre.

    STRICT FORMAT:
    CURRENT STATUS:
    IMMEDIATE ACTION:
    TOTAL VOLUME:
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

@app.post("/predict")
def predict(data: SensorData):

    advice = generate_advice(data)

    # Save latest reading
    global latest_data
    latest_data = {
        "temperature": data.temperature,
        "humidity": data.humidity,
        "soil": data.soil,
        "advice": advice
    }

    return {"advice": advice}