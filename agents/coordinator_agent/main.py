"""Coordinator Agent - Orchestrates trip planning across all agents."""
import os
import asyncio
import httpx
from dotenv import load_dotenv
from agentfield import Agent, AIConfig

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.models import TripPlanRequest, TripPlan

load_dotenv()

# Direct agent URLs for local development
WEATHER_AGENT_URL = os.getenv("WEATHER_AGENT_URL", "http://localhost:8001")
SEARCH_AGENT_URL = os.getenv("SEARCH_AGENT_URL", "http://localhost:8002")
LOCATION_AGENT_URL = os.getenv("LOCATION_AGENT_URL", "http://localhost:8003")


async def call_agent(base_url: str, skill: str, params: dict) -> dict:
    """Call another agent directly via HTTP."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{base_url}/skills/{skill}", json=params)
        response.raise_for_status()
        return response.json()

# Initialize the Agentfield agent
app = Agent(
    node_id="coordinator-agent",
    ai_config=AIConfig(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0.3
    ),
    enable_did=False
)


@app.skill()
async def plan_trip(
    destination: str,
    start_date: str,
    end_date: str,
    interests: list[str] = None,
    budget: str = ""
) -> dict:
    """
    Plan a complete trip by coordinating all agents.
    
    Args:
        destination: Trip destination
        start_date: Trip start date (YYYY-MM-DD)
        end_date: Trip end date (YYYY-MM-DD)
        interests: List of interests (e.g., ["hiking", "food", "history"])
        budget: Budget level (luxury, mid-range, budget)
    
    Returns:
        Complete trip plan with weather, attractions, and recommendations
    """
    interests = interests or []
    
    trip_plan = {
        "destination": destination,
        "start_date": start_date,
        "end_date": end_date,
        "interests": interests,
        "budget": budget,
        "location": None,
        "weather_forecast": None,
        "attractions": [],
        "restaurants": [],
        "accommodations": [],
        "tips": [],
        "summary": ""
    }
    
    # Step 1: Get location data
    try:
        location_result = await call_agent(
            LOCATION_AGENT_URL,
            "geocode_address",
            {"address": destination}
        )
        
        if "error" not in location_result:
            trip_plan["location"] = location_result
            latitude = location_result.get("latitude")
            longitude = location_result.get("longitude")
        else:
            latitude = None
            longitude = None
    except Exception as e:
        latitude = None
        longitude = None
        trip_plan["errors"] = trip_plan.get("errors", []) + [f"Location lookup failed: {str(e)}"]
    
    # Step 2: Get weather forecast (if we have coordinates)
    if latitude and longitude:
        try:
            weather_result = await call_agent(
                WEATHER_AGENT_URL,
                "get_weather_forecast",
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "location_name": destination,
                    "days": 14
                }
            )
            trip_plan["weather_forecast"] = weather_result
        except Exception as e:
            trip_plan["errors"] = trip_plan.get("errors", []) + [f"Weather lookup failed: {str(e)}"]
    
    # Step 3: Search for attractions, restaurants, and accommodations in parallel
    search_tasks = []
    
    # Search attractions
    try:
        attractions_result = await call_agent(
            SEARCH_AGENT_URL,
            "search_attractions",
            {"destination": destination, "num_results": 5}
        )
        trip_plan["attractions"] = attractions_result.get("results", [])
    except Exception as e:
        trip_plan["errors"] = trip_plan.get("errors", []) + [f"Attractions search failed: {str(e)}"]
    
    # Search restaurants
    try:
        restaurants_result = await call_agent(
            SEARCH_AGENT_URL,
            "search_restaurants",
            {"destination": destination, "num_results": 5}
        )
        trip_plan["restaurants"] = restaurants_result.get("results", [])
    except Exception as e:
        trip_plan["errors"] = trip_plan.get("errors", []) + [f"Restaurants search failed: {str(e)}"]
    
    # Search accommodations
    try:
        accommodations_result = await call_agent(
            SEARCH_AGENT_URL,
            "search_accommodations",
            {"destination": destination, "budget": budget, "num_results": 5}
        )
        trip_plan["accommodations"] = accommodations_result.get("results", [])
    except Exception as e:
        trip_plan["errors"] = trip_plan.get("errors", []) + [f"Accommodations search failed: {str(e)}"]
    
    # Search travel tips
    try:
        tips_result = await call_agent(
            SEARCH_AGENT_URL,
            "search_travel_tips",
            {"destination": destination, "num_results": 5}
        )
        trip_plan["tips"] = [r.get("snippet", "") for r in tips_result.get("results", [])]
    except Exception as e:
        trip_plan["errors"] = trip_plan.get("errors", []) + [f"Tips search failed: {str(e)}"]
    
    return trip_plan


@app.skill()
async def get_weather_for_destination(destination: str, days: int = 7) -> dict:
    """
    Get weather forecast for a destination by name.
    
    Args:
        destination: Destination name
        days: Number of days to forecast
    
    Returns:
        Weather forecast
    """
    # First geocode the destination
    location_result = await call_agent(
        LOCATION_AGENT_URL,
        "geocode_address",
        {"address": destination}
    )
    
    if "error" in location_result:
        return {"error": location_result["error"]}
    
    # Then get the weather
    weather_result = await call_agent(
        WEATHER_AGENT_URL,
        "get_weather_forecast",
        {
            "latitude": location_result["latitude"],
            "longitude": location_result["longitude"],
            "location_name": destination,
            "days": days
        }
    )
    
    return weather_result


@app.skill()
async def search_destination(destination: str, query_type: str = "attractions") -> dict:
    """
    Search for information about a destination.
    
    Args:
        destination: Destination name
        query_type: Type of search (attractions, restaurants, accommodations, tips)
    
    Returns:
        Search results
    """
    skill_map = {
        "attractions": "search_attractions",
        "restaurants": "search_restaurants",
        "accommodations": "search_accommodations",
        "tips": "search_travel_tips"
    }
    
    skill_name = skill_map.get(query_type, "search_attractions")
    
    result = await call_agent(
        SEARCH_AGENT_URL,
        skill_name,
        {"destination": destination, "num_results": 5}
    )
    
    return result


@app.reasoner()
async def generate_trip_summary(trip_plan: dict) -> dict:
    """
    Generate a comprehensive trip summary with personalized recommendations.
    
    Args:
        trip_plan: Complete trip plan data
    
    Returns:
        Trip summary with recommendations
    """
    result = await app.ai(
        system="""You are an expert travel planner. Create a comprehensive, 
        personalized trip summary based on the collected data. Be specific, 
        practical, and enthusiastic about the destination.""",
        user=f"""Trip plan data: {trip_plan}
        
Create a trip summary including:
1. Destination overview (2-3 sentences)
2. Weather summary and what to pack
3. Top 3 must-see attractions with brief descriptions
4. Top 3 restaurant recommendations
5. Accommodation recommendations based on budget
6. Day-by-day activity suggestions
7. Important tips and things to know
8. Estimated budget breakdown (if applicable)""",
        schema={
            "type": "object",
            "properties": {
                "destination_overview": {"type": "string"},
                "weather_summary": {"type": "string"},
                "packing_recommendations": {"type": "array", "items": {"type": "string"}},
                "must_see_attractions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "recommended_time": {"type": "string"}
                        }
                    }
                },
                "restaurant_recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "cuisine": {"type": "string"},
                            "price_range": {"type": "string"}
                        }
                    }
                },
                "accommodation_tips": {"type": "string"},
                "daily_itinerary": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "day": {"type": "string"},
                            "activities": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "important_tips": {"type": "array", "items": {"type": "string"}},
                "budget_estimate": {"type": "string"}
            },
            "required": ["destination_overview", "must_see_attractions", "important_tips"]
        }
    )
    
    return result


@app.skill()
async def create_full_trip_plan(
    destination: str,
    start_date: str,
    end_date: str,
    interests: list[str] = None,
    budget: str = ""
) -> dict:
    """
    Create a complete trip plan with AI-generated summary.
    
    Args:
        destination: Trip destination
        start_date: Trip start date (YYYY-MM-DD)
        end_date: Trip end date (YYYY-MM-DD)
        interests: List of interests
        budget: Budget level
    
    Returns:
        Complete trip plan with AI summary
    """
    # First gather all the data
    trip_plan = await plan_trip(
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        interests=interests,
        budget=budget
    )
    
    # Then generate an AI summary
    summary = await generate_trip_summary(trip_plan)
    trip_plan["ai_summary"] = summary
    
    return trip_plan


if __name__ == "__main__":
    app.run(port=8000)
