import pandas as pd
import requests


TOP_UKRAINIAN_CITIES = [
    {
        "name": "Київ",
        "country": "Україна",
        "country_code": "UA",
        "admin_area": "Київ",
        "latitude": 50.45466,
        "longitude": 30.5238,
    },
    {
        "name": "Харків",
        "country": "Україна",
        "country_code": "UA",
        "admin_area": "Харківська область",
        "latitude": 49.98081,
        "longitude": 36.25272,
    },
    {
        "name": "Одеса",
        "country": "Україна",
        "country_code": "UA",
        "admin_area": "Одеська область",
        "latitude": 46.47747,
        "longitude": 30.73262,
    },
    {
        "name": "Дніпро",
        "country": "Україна",
        "country_code": "UA",
        "admin_area": "Дніпропетровська область",
        "latitude": 48.46664,
        "longitude": 35.04066,
    },
    {
        "name": "Львів",
        "country": "Україна",
        "country_code": "UA",
        "admin_area": "Львівська область",
        "latitude": 49.83826,
        "longitude": 24.02324,
    },
]


def search_ukrainian_locations(query, limit=5):
    query = query.strip()
    if len(query) < 2:
        return TOP_UKRAINIAN_CITIES[:limit]

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": query,
        "count": 10,
        "language": "uk",
        "format": "json",
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    ukrainian_results = [
        {
            "name": item["name"],
            "country": item.get("country", ""),
            "country_code": item.get("country_code", ""),
            "admin_area": item.get("admin1", ""),
            "latitude": item["latitude"],
            "longitude": item["longitude"],
        }
        for item in data.get("results", [])
        if item.get("country_code") == "UA"
    ]

    return ukrainian_results[:limit]


def format_location(location):
    parts = [
        location.get("name", ""),
        location.get("admin_area", ""),
    ]
    return ", ".join(part for part in parts if part)


def get_city_coordinates(city):
    ukrainian_results = search_ukrainian_locations(city, limit=1)

    if not ukrainian_results:
        raise ValueError(
            f"Населений пункт '{city}' в Україні не знайдено. "
            "Спробуй написати назву українською або англійською."
        )

    return ukrainian_results[0]


def load_weather_data(city, start_date, end_date, location=None):
    if location is None:
        location = get_city_coordinates(city)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ],
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "daily" not in data:
        reason = data.get("reason", "Open-Meteo не повернув погодні дані.")
        raise ValueError(reason)

    daily = data["daily"]

    df = pd.DataFrame({
        "date": daily["time"],
        "temperature_max": daily["temperature_2m_max"],
        "temperature_min": daily["temperature_2m_min"],
        "precipitation_sum": daily["precipitation_sum"],
        "wind_speed": daily["wind_speed_10m_max"],
    })

    df.attrs["location"] = location
    df.to_csv("weather_daily.csv", index=False)

    return df
