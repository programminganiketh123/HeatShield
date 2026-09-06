from fastapi import FastAPI
from src.risk_engine import calculate_risk

app = FastAPI(
    title="HeatShield API",
    description="Extreme Heat and Human Thermal Stress API",
    version="1.0",
)

@app.get("/")
def home():
    return {"system": "HeatShield", "status": "online"}

@app.get("/risk/hyderabad")
def hyderabad_risk():
    result = calculate_risk(
        temperature=41,
        humidity=68,
        wind_speed=6,
        solar_radiation=850,
        population_density=16000,
        elderly_percentage=18,
        outdoor_worker_percentage=32,
        healthcare_access=40,
    )
    return {"location": "Hyderabad", **result}
