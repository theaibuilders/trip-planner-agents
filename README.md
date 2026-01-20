# Trip Planner Agent Network

A multi-agent trip planning system built with [Agentfield](https://agentfield.ai) that combines weather forecasting, web search, and location services to create comprehensive travel plans.

## Architecture

```
                    ┌─────────────────────┐
                    │  Coordinator Agent  │
                    │    (Port 8000)      │
                    └─────────┬───────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Weather Agent  │  │  Search Agent   │  │ Location Agent  │
│   (Port 8001)   │  │   (Port 8002)   │  │   (Port 8003)   │
│                 │  │                 │  │                 │
│  Open Meteo API │  │ Bright Data SERP│  │ Google Geo API  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Agents

### 1. Weather Agent (Port 8001)
Uses the Open Meteo API to provide weather forecasts.

**Skills:**
- `get_weather_forecast`: Get multi-day weather forecast for coordinates
- `get_current_weather`: Get current weather conditions

**Reasoners:**
- `analyze_weather_for_trip`: AI analysis of weather impact on planned activities

### 2. Search Agent (Port 8002)
Uses Bright Data SERP API for web searches.

**Skills:**
- `search_web`: General web search
- `search_attractions`: Find tourist attractions
- `search_restaurants`: Find restaurants and dining options
- `search_accommodations`: Find hotels and lodging
- `search_travel_tips`: Find travel guides and tips
- `search_events`: Find events across lu.ma, meetup.com, eventbrite
- `search_tech_events`: Find tech/AI/startup events and meetups

**Reasoners:**
- `summarize_search_results`: AI summarization of search results

### 3. Location Agent (Port 8003)
Uses Google Geocoding and Places APIs for location data.

**Skills:**
- `geocode_address`: Convert address to coordinates
- `reverse_geocode`: Convert coordinates to address
- `search_nearby_places`: Find nearby points of interest
- `calculate_distance`: Calculate distance between two points

**Reasoners:**
- `analyze_location_for_trip`: AI analysis of location characteristics

### 4. Coordinator Agent (Port 8000)
Orchestrates all other agents to create comprehensive trip plans.

**Skills:**
- `plan_trip`: Create a complete trip plan
- `get_weather_for_destination`: Get weather by destination name
- `search_destination`: Search for destination information
- `create_full_trip_plan`: Complete trip plan with AI summary

**Reasoners:**
- `generate_trip_summary`: AI-generated trip summary and recommendations

## Prerequisites

- Python 3.11+
- [Agentfield CLI](https://agentfield.ai/docs/quick-start)
- API Keys:
  - OpenAI API key (for AI reasoning)
  - Bright Data API key (for web search)
  - Google Cloud API key (for geocoding/places)

## Installation

### 1. Install Agentfield CLI

```bash
curl -sSf https://agentfield.ai/get | sh
```

Verify installation:
```bash
af --version
```

### 2. Clone and Setup

```bash
cd trip-planner-agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
```

Required environment variables:
```
OPENAI_API_KEY=your_openai_api_key
BRIGHT_DATA_API_KEY=your_bright_data_api_key
GOOGLE_GEO_API_KEY=your_google_api_key
```

## Running the Agents

### Option 1: Using the Startup Script (Recommended)

```bash
chmod +x start_agents.sh
./start_agents.sh
```

This script will:
- Check and install dependencies automatically
- Set the required PYTHONPATH
- Start all agents in the correct order

### Option 2: Manual Startup

```bash
# Set PYTHONPATH and start each agent
export PYTHONPATH=$PWD:$PYTHONPATH

python agents/weather_agent/main.py &
python agents/search_agent/main.py &
python agents/location_agent/main.py &
python agents/coordinator_agent/main.py &
```

### Option 3: Using Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### Stopping Agents

```bash
chmod +x stop_agents.sh
./stop_agents.sh
```

## API Usage

### Create a Trip Plan

```bash
curl -X POST http://localhost:8000/skills/plan_trip \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "San Francisco Bay Area, California",
    "start_date": "2026-01-22",
    "end_date": "2025-02-01",
    "interests": ["ai or tech events", "korean bbq", "nice coffee shops"],
    "budget": "2000 USD"
  }'
```

### Get Weather Forecast

```bash
curl -X POST http://localhost:8000/skills/get_weather_for_destination \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Tokyo, Japan",
    "days": 7
  }'
```

### Search for Attractions

```bash
curl -X POST http://localhost:8000/skills/search_destination \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Barcelona, Spain",
    "query_type": "attractions"
  }'
```

### Create Full Trip Plan with AI Summary

```bash
curl -X POST http://localhost:8000/skills/create_full_trip_plan \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Paris, France",
    "start_date": "2025-06-01",
    "end_date": "2025-06-07",
    "interests": ["art", "food", "history"],
    "budget": "mid-range"
  }'
```

### Direct Agent Calls

You can also call individual agents directly:

```bash
# Weather Agent
curl -X POST http://localhost:8001/skills/get_current_weather \
  -H "Content-Type: application/json" \
  -d '{"latitude": 48.8566, "longitude": 2.3522, "location_name": "Paris"}'

# Location Agent
curl -X POST http://localhost:8003/skills/geocode_address \
  -H "Content-Type: application/json" \
  -d '{"address": "Eiffel Tower, Paris"}'

# Search Agent
curl -X POST http://localhost:8002/skills/search_attractions \
  -H "Content-Type: application/json" \
  -d '{"destination": "Rome, Italy", "num_results": 5}'

# Search for Events
curl -X POST http://localhost:8002/skills/search_events \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "San Francisco",
    "event_type": "food",
    "date_range": "this week",
    "num_results": 5
  }'

# Search for Tech/AI Events
curl -X POST http://localhost:8002/skills/search_tech_events \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "San Francisco Bay Area",
    "topic": "AI",
    "date_range": "January 2026",
    "num_results": 5
  }'
```

## Project Structure

```
trip-planner-agents/
├── agents/
│   ├── coordinator_agent/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── weather_agent/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── search_agent/
│   │   ├── __init__.py
│   │   └── main.py
│   └── location_agent/
│       ├── __init__.py
│       └── main.py
├── common/
│   ├── __init__.py
│   └── models.py
├── agentfield.yaml
├── docker-compose.yml
├── Dockerfile.agent
├── requirements.txt
├── start_agents.sh
├── stop_agents.sh
├── .env.example
└── README.md
```

## API Keys Setup

### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create an API key in your account settings
3. Add to `.env` as `OPENAI_API_KEY`

### Bright Data SERP API Key
1. Sign up at [Bright Data](https://brightdata.com/)
2. Create a SERP API zone in your dashboard
3. Copy the API key and zone name
4. Add to `.env` as `BRIGHT_DATA_API_KEY` and `BRIGHT_DATA_ZONE`

### Google API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable:
   - Geocoding API
   - Places API
3. Create an API key with appropriate restrictions
4. Add to `.env` as `GOOGLE_GEO_API_KEY`

## Development

### Adding New Skills

To add a new skill to an agent:

```python
@app.skill()
async def my_new_skill(param1: str, param2: int) -> dict:
    """
    Description of what this skill does.
    
    Args:
        param1: Description
        param2: Description
    
    Returns:
        Result description
    """
    # Implementation
    return {"result": "value"}
```

### Adding New Reasoners

To add AI-powered reasoning:

```python
@app.reasoner()
async def my_reasoner(data: dict) -> dict:
    """AI-powered analysis."""
    result = await app.ai(
        system="System prompt",
        user=f"User prompt with {data}",
        schema={
            "type": "object",
            "properties": {
                "analysis": {"type": "string"}
            }
        }
    )
    return result
```

## Troubleshooting

### Agents not connecting
- Check that ports 8000-8003 are available
- Verify agents are running: `lsof -i :8000` (repeat for 8001-8003)
- Use `./stop_agents.sh` to clean up before restarting

### DID Registration Errors
- All agents are configured with `enable_did=False` for standalone mode
- This is expected for local development without an Agentfield orchestration server

### API errors
- Verify all API keys are set correctly in `.env`
- Check API key permissions and quotas
- Review agent logs for specific error messages

### Import errors / ModuleNotFoundError
- Use `./start_agents.sh` which automatically installs dependencies
- Or manually run: `pip install -r requirements.txt`
- Ensure PYTHONPATH includes the project root: `export PYTHONPATH=$PWD:$PYTHONPATH`

### Port already in use
- Run `./stop_agents.sh` to stop all agents
- Or manually kill: `lsof -ti:8000 | xargs kill -9`

## License

MIT License
