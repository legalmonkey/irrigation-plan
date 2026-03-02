from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os



from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ----------------- STATE STORAGE -----------------
latest_data = {}
reading_counter = 0
last_advice = "Waiting for first AI cycle..."

# -------------------------------------------------

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
    Provide irrigation recommendation.
    """

    response = model.generate_content(prompt)

    if not response or not response.text:
        return "AI returned empty response."

    return response.text

@app.post("/predict")
def predict(data: SensorData):

    global reading_counter, latest_data, last_advice

    reading_counter += 1

    # Only call Gemini every 12 readings
    if reading_counter >= 12:
        try:
            last_advice = generate_advice(data)
        except Exception as e:
            last_advice = f"AI Error: {str(e)}"

        reading_counter = 0  # reset counter

    # Always update sensor data
    latest_data = {
        "temperature": data.temperature,
        "humidity": data.humidity,
        "soil": data.soil,
        "advice": last_advice,
        "reading_count": reading_counter
    }

    return {"advice": last_advice}