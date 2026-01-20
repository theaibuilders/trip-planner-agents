"""Pydantic models for trip planner agents."""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class WeatherData(BaseModel):
    """Weather data for a location."""
    location: str
    date: str
    temperature_max: float = Field(description="Maximum temperature in Celsius")
    temperature_min: float = Field(description="Minimum temperature in Celsius")
    precipitation_probability: float = Field(description="Probability of precipitation in %")
    weather_code: int = Field(description="WMO weather code")
    weather_description: str = Field(description="Human-readable weather description")


class WeatherForecast(BaseModel):
    """Multi-day weather forecast."""
    location: str
    latitude: float
    longitude: float
    forecast: list[WeatherData]


class LocationData(BaseModel):
    """Geocoded location data."""
    address: str
    latitude: float
    longitude: float
    place_id: Optional[str] = None
    formatted_address: str
    country: Optional[str] = None
    city: Optional[str] = None


class PlaceDetails(BaseModel):
    """Details about a place."""
    name: str
    address: str
    latitude: float
    longitude: float
    types: list[str] = []
    rating: Optional[float] = None


class SearchResult(BaseModel):
    """Single search result."""
    title: str
    url: str
    snippet: str


class SearchResults(BaseModel):
    """Web search results."""
    query: str
    results: list[SearchResult]


class TripPlanRequest(BaseModel):
    """Request for trip planning."""
    destination: str
    start_date: str
    end_date: str
    interests: list[str] = Field(default_factory=list)
    budget: Optional[str] = None


class TripPlan(BaseModel):
    """Complete trip plan."""
    destination: str
    location: Optional[LocationData] = None
    weather_forecast: Optional[WeatherForecast] = None
    attractions: list[SearchResult] = []
    restaurants: list[SearchResult] = []
    accommodations: list[SearchResult] = []
    tips: list[str] = []
    summary: str = ""
