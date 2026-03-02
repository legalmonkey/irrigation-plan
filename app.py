from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
import os

app = FastAPI()

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

class SensorData(BaseModel):
    temperature: float
    humidity: float
    soil: float
    crop: str
    location: str
    farm_size: float

def generate_advice(data: SensorData):

    prompt = f"""
    You are a Real-Time Irrigation AI for a farmer in {data.location}.
    Crop: {data.crop} | Farm Size: {data.farm_size} Acres.

    LATEST SENSOR READING:
    Temp: {data.temperature}
    Humidity: {data.humidity}
    Soil: {data.soil}

    1mm = 4046 Liters per Acre.
    Give irrigation recommendation.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

@app.post("/predict")
def predict(data: SensorData):
    advice = generate_advice(data)
    return {"advice": advice}