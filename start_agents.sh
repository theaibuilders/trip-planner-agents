#!/bin/bash
# Start all agents for local development

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Trip Planner Agent Network${NC}"
echo "========================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found. Copying from .env.example${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env with your API keys${NC}"
fi

# Source environment variables
export $(grep -v '^#' .env | xargs)

# Set PYTHONPATH to include project root
export PYTHONPATH=$PWD:$PYTHONPATH

# Install requirements if needed
echo -e "${GREEN}Checking dependencies...${NC}"
pip install -q -r requirements.txt 2>/dev/null

# Function to start an agent in the background
start_agent() {
    local name=$1
    local port=$2
    local path=$3
    
    echo -e "${GREEN}Starting ${name} on port ${port}...${NC}"
    python ${path}/main.py &
    sleep 2
}

# Start agents in order
echo ""
echo "Starting individual agents..."
echo ""

start_agent "Weather Agent" 8001 "agents/weather_agent"
start_agent "Search Agent" 8002 "agents/search_agent"
start_agent "Location Agent" 8003 "agents/location_agent"
start_agent "Coordinator Agent" 8000 "agents/coordinator_agent"

echo ""
echo -e "${GREEN}All agents started!${NC}"
echo ""
echo "Agent Endpoints:"
echo "  - Coordinator: http://localhost:8000"
echo "  - Weather:     http://localhost:8001"
echo "  - Search:      http://localhost:8002"
echo "  - Location:    http://localhost:8003"
echo ""
echo "Press Ctrl+C to stop all agents"

# Wait for all background processes
wait
