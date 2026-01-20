"""Test script to verify imports and basic functionality."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_models():
    """Test that models can be imported and instantiated."""
    from common.models import (
        WeatherData, WeatherForecast, LocationData, 
        PlaceDetails, SearchResult, SearchResults,
        TripPlanRequest, TripPlan
    )
    
    # Test WeatherData
    weather = WeatherData(
        location="Paris",
        date="2025-06-01",
        temperature_max=25.0,
        temperature_min=15.0,
        precipitation_probability=10.0,
        weather_code=1,
        weather_description="Mainly clear"
    )
    assert weather.location == "Paris"
    print("WeatherData: OK")
    
    # Test LocationData
    location = LocationData(
        address="Paris, France",
        latitude=48.8566,
        longitude=2.3522,
        formatted_address="Paris, France"
    )
    assert location.latitude == 48.8566
    print("LocationData: OK")
    
    # Test SearchResult
    result = SearchResult(
        title="Test Result",
        url="https://example.com",
        snippet="This is a test"
    )
    assert result.title == "Test Result"
    print("SearchResult: OK")
    
    # Test TripPlanRequest
    request = TripPlanRequest(
        destination="Tokyo, Japan",
        start_date="2025-07-01",
        end_date="2025-07-07",
        interests=["food", "culture"]
    )
    assert request.destination == "Tokyo, Japan"
    print("TripPlanRequest: OK")
    
    print("\nAll model tests passed!")


def test_weather_codes():
    """Test weather code dictionary."""
    # Import from weather agent without running
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "weather_agent", 
        "agents/weather_agent/main.py"
    )
    
    # We can't fully load due to agentfield dependency
    # but we can read and parse the file
    with open("agents/weather_agent/main.py", "r") as f:
        content = f.read()
        assert "WEATHER_CODES" in content
        assert "Clear sky" in content
        assert "Thunderstorm" in content
    
    print("Weather codes: OK")


if __name__ == "__main__":
    print("Running tests...\n")
    
    print("Testing models...")
    test_models()
    
    print("\nTesting weather agent structure...")
    test_weather_codes()
    
    print("\n" + "="*50)
    print("All tests passed successfully!")
    print("="*50)
    print("\nNote: Full integration tests require:")
    print("  - Agentfield SDK installed (pip install agentfield)")
    print("  - API keys configured in .env")
    print("  - Agentfield server running (af server)")
