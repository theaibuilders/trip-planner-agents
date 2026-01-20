"""Weather Agent - Uses Open Meteo API for weather forecasts."""
import os
import httpx
from datetime import datetime, timedelta
from dotenv import load_dotenv
from agentfield import Agent, AIConfig

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.models import WeatherData, WeatherForecast

load_dotenv()

# Weather code descriptions based on WMO codes
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# Initialize the Agentfield agent
app = Agent(
    node_id="weather-agent",
    ai_config=AIConfig(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0.2
    ),
    enable_did=False
)


@app.skill()
async def get_weather_forecast(
    latitude: float,
    longitude: float,
    location_name: str,
    days: int = 7
) -> dict:
    """
    Get weather forecast for a location using Open Meteo API.
    
    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
        location_name: Human-readable name of the location
        days: Number of days to forecast (max 16)
    
    Returns:
        Weather forecast data
    """
    days = min(days, 16)  # Open Meteo supports up to 16 days
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "weather_code"
        ],
        "timezone": "auto",
        "forecast_days": days
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    precip_prob = daily.get("precipitation_probability_max", [])
    weather_codes = daily.get("weather_code", [])
    
    forecast_items = []
    for i in range(len(dates)):
        code = weather_codes[i] if i < len(weather_codes) else 0
        forecast_items.append(WeatherData(
            location=location_name,
            date=dates[i],
            temperature_max=temp_max[i] if i < len(temp_max) else 0,
            temperature_min=temp_min[i] if i < len(temp_min) else 0,
            precipitation_probability=precip_prob[i] if i < len(precip_prob) else 0,
            weather_code=code,
            weather_description=WEATHER_CODES.get(code, "Unknown")
        ))
    
    forecast = WeatherForecast(
        location=location_name,
        latitude=latitude,
        longitude=longitude,
        forecast=forecast_items
    )
    
    return forecast.model_dump()


@app.skill()
async def get_current_weather(latitude: float, longitude: float, location_name: str) -> dict:
    """
    Get current weather for a location using Open Meteo API.
    
    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
        location_name: Human-readable name of the location
    
    Returns:
        Current weather data
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "weather_code",
            "wind_speed_10m"
        ],
        "timezone": "auto"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    
    current = data.get("current", {})
    code = current.get("weather_code", 0)
    
    return {
        "location": location_name,
        "latitude": latitude,
        "longitude": longitude,
        "temperature": current.get("temperature_2m"),
        "apparent_temperature": current.get("apparent_temperature"),
        "humidity": current.get("relative_humidity_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "weather_code": code,
        "weather_description": WEATHER_CODES.get(code, "Unknown")
    }


@app.reasoner()
async def analyze_weather_for_trip(
    forecast_data: dict,
    activities: list[str]
) -> dict:
    """
    Analyze weather forecast and provide recommendations for planned activities.
    
    Args:
        forecast_data: Weather forecast data from get_weather_forecast
        activities: List of planned activities
    
    Returns:
        Weather analysis and recommendations
    """
    activities_str = ", ".join(activities) if activities else "general sightseeing"
    
    result = await app.ai(
        system="""You are a weather analyst helping plan trips. 
        Analyze the weather forecast and provide practical recommendations 
        for the planned activities. Be specific about which days are best 
        for which activities.""",
        user=f"""Weather forecast: {forecast_data}
        
Planned activities: {activities_str}

Provide:
1. Overall weather summary for the trip
2. Best days for outdoor activities
3. Days to avoid for outdoor activities
4. Packing recommendations based on weather
5. Activity-specific weather considerations""",
        schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "best_outdoor_days": {"type": "array", "items": {"type": "string"}},
                "avoid_outdoor_days": {"type": "array", "items": {"type": "string"}},
                "packing_list": {"type": "array", "items": {"type": "string"}},
                "activity_recommendations": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["summary", "packing_list", "activity_recommendations"]
        }
    )
    
    return result


if __name__ == "__main__":
    app.run(port=8001)
