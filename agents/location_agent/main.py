"""Location Agent - Uses Google Geocoding API for location data."""
import os
import httpx
from urllib.parse import quote_plus
from dotenv import load_dotenv
from agentfield import Agent, AIConfig

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.models import LocationData, PlaceDetails

load_dotenv()

# Initialize the Agentfield agent
app = Agent(
    node_id="location-agent",
    ai_config=AIConfig(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0.2
    ),
    enable_did=False
)

GOOGLE_GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GOOGLE_PLACES_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
GOOGLE_PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"


@app.skill()
async def geocode_address(address: str) -> dict:
    """
    Convert an address to geographic coordinates using Google Geocoding API.
    
    Args:
        address: Address or place name to geocode
    
    Returns:
        Location data with coordinates and formatted address
    """
    api_key = os.getenv("GOOGLE_GEO_API_KEY")
    
    if not api_key:
        return {
            "error": "GOOGLE_GEO_API_KEY environment variable not set",
            "address": address
        }
    
    params = {
        "address": address,
        "key": api_key
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(GOOGLE_GEOCODING_URL, params=params)
            response.raise_for_status()
            data = response.json()
        
        if data.get("status") != "OK":
            return {
                "error": f"Geocoding failed: {data.get('status')}",
                "address": address
            }
        
        result = data["results"][0]
        location = result["geometry"]["location"]
        
        # Extract address components
        country = None
        city = None
        for component in result.get("address_components", []):
            types = component.get("types", [])
            if "country" in types:
                country = component.get("long_name")
            if "locality" in types:
                city = component.get("long_name")
        
        location_data = LocationData(
            address=address,
            latitude=location["lat"],
            longitude=location["lng"],
            place_id=result.get("place_id"),
            formatted_address=result.get("formatted_address", address),
            country=country,
            city=city
        )
        
        return location_data.model_dump()
    
    except httpx.HTTPStatusError as e:
        return {
            "error": f"HTTP error: {e.response.status_code}",
            "address": address
        }
    except Exception as e:
        return {
            "error": str(e),
            "address": address
        }


@app.skill()
async def reverse_geocode(latitude: float, longitude: float) -> dict:
    """
    Convert coordinates to an address using Google Geocoding API.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
    
    Returns:
        Location data with address information
    """
    api_key = os.getenv("GOOGLE_GEO_API_KEY")
    
    if not api_key:
        return {
            "error": "GOOGLE_GEO_API_KEY environment variable not set"
        }
    
    params = {
        "latlng": f"{latitude},{longitude}",
        "key": api_key
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(GOOGLE_GEOCODING_URL, params=params)
            response.raise_for_status()
            data = response.json()
        
        if data.get("status") != "OK":
            return {
                "error": f"Reverse geocoding failed: {data.get('status')}"
            }
        
        result = data["results"][0]
        
        # Extract address components
        country = None
        city = None
        for component in result.get("address_components", []):
            types = component.get("types", [])
            if "country" in types:
                country = component.get("long_name")
            if "locality" in types:
                city = component.get("long_name")
        
        location_data = LocationData(
            address=result.get("formatted_address", ""),
            latitude=latitude,
            longitude=longitude,
            place_id=result.get("place_id"),
            formatted_address=result.get("formatted_address", ""),
            country=country,
            city=city
        )
        
        return location_data.model_dump()
    
    except Exception as e:
        return {"error": str(e)}


@app.skill()
async def search_nearby_places(
    latitude: float,
    longitude: float,
    place_type: str,
    radius: int = 5000
) -> dict:
    """
    Search for nearby places using Google Places API.
    
    Args:
        latitude: Center latitude
        longitude: Center longitude
        place_type: Type of place (restaurant, hotel, tourist_attraction, etc.)
        radius: Search radius in meters (default 5000)
    
    Returns:
        List of nearby places
    """
    api_key = os.getenv("GOOGLE_GEO_API_KEY")
    
    if not api_key:
        return {
            "error": "GOOGLE_GEO_API_KEY environment variable not set",
            "places": []
        }
    
    params = {
        "query": place_type,
        "location": f"{latitude},{longitude}",
        "radius": radius,
        "key": api_key
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(GOOGLE_PLACES_URL, params=params)
            response.raise_for_status()
            data = response.json()
        
        if data.get("status") not in ["OK", "ZERO_RESULTS"]:
            return {
                "error": f"Places search failed: {data.get('status')}",
                "places": []
            }
        
        places = []
        for result in data.get("results", [])[:10]:
            location = result.get("geometry", {}).get("location", {})
            places.append(PlaceDetails(
                name=result.get("name", ""),
                address=result.get("formatted_address", ""),
                latitude=location.get("lat", 0),
                longitude=location.get("lng", 0),
                types=result.get("types", []),
                rating=result.get("rating")
            ).model_dump())
        
        return {
            "query": place_type,
            "center": {"latitude": latitude, "longitude": longitude},
            "radius": radius,
            "places": places
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "places": []
        }


@app.skill()
async def calculate_distance(
    origin_lat: float,
    origin_lng: float,
    dest_lat: float,
    dest_lng: float
) -> dict:
    """
    Calculate approximate distance between two points (Haversine formula).
    
    Args:
        origin_lat: Origin latitude
        origin_lng: Origin longitude
        dest_lat: Destination latitude
        dest_lng: Destination longitude
    
    Returns:
        Distance in kilometers
    """
    import math
    
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(origin_lat)
    lat2_rad = math.radians(dest_lat)
    delta_lat = math.radians(dest_lat - origin_lat)
    delta_lng = math.radians(dest_lng - origin_lng)
    
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lng / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance_km = R * c
    
    return {
        "origin": {"latitude": origin_lat, "longitude": origin_lng},
        "destination": {"latitude": dest_lat, "longitude": dest_lng},
        "distance_km": round(distance_km, 2),
        "distance_miles": round(distance_km * 0.621371, 2)
    }


@app.reasoner()
async def analyze_location_for_trip(
    location_data: dict,
    trip_type: str = "leisure"
) -> dict:
    """
    Analyze a location and provide trip planning insights.
    
    Args:
        location_data: Geocoded location data
        trip_type: Type of trip (leisure, business, adventure, etc.)
    
    Returns:
        Location analysis for trip planning
    """
    result = await app.ai(
        system="""You are a travel location analyst. 
        Analyze the location and provide practical insights for trip planning.
        Consider geographic features, climate zones, and regional characteristics.""",
        user=f"""Location data: {location_data}
        Trip type: {trip_type}
        
Provide:
1. Geographic context (region, terrain, climate zone)
2. Best time to visit considerations
3. Local transportation options typically available
4. Nearby major cities or attractions
5. Cultural or practical considerations""",
        schema={
            "type": "object",
            "properties": {
                "geographic_context": {"type": "string"},
                "best_time_to_visit": {"type": "string"},
                "transportation_options": {"type": "array", "items": {"type": "string"}},
                "nearby_highlights": {"type": "array", "items": {"type": "string"}},
                "practical_tips": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["geographic_context", "best_time_to_visit"]
        }
    )
    
    return result


if __name__ == "__main__":
    app.run(port=8003)
