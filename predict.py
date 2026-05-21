import joblib
import pandas as pd
import requests

from data_loader import get_city_coordinates
from train_model import prepare_features


def predict_weather(city, location=None):
    if location is None:
        location = get_city_coordinates(city)
    latitude = location["latitude"]
    longitude = location["longitude"]

    model_data = joblib.load("models/rain_model.pkl")
    model = model_data["model"]
    selected_features = model_data["selected_features"]
    imputation_values = model_data["imputation_values"]

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "wind_speed_10m_max",
        ],
        "forecast_days": 5,
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    daily = data["daily"]

    df = pd.DataFrame({
        "date": daily["time"],
        "temperature_max": daily["temperature_2m_max"],
        "temperature_min": daily["temperature_2m_min"],
        "wind_speed": daily["wind_speed_10m_max"],
    })

    X_forecast = prepare_features(df)
    X_forecast = X_forecast.fillna(imputation_values)
    X_forecast = X_forecast[selected_features]

    predictions = model.predict(X_forecast)
    probabilities = model.predict_proba(X_forecast)[:, 1]

    df["prediction"] = predictions
    df["probability"] = probabilities

    return df
