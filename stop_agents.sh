#!/bin/bash
# Stop all agents for local development

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping Trip Planner Agent Network${NC}"
echo "========================================"

# Function to stop agents by port
stop_agent_by_port() {
    local name=$1
    local port=$2
    
    pid=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pid" ]; then
        echo -e "${GREEN}Stopping ${name} (PID: ${pid}) on port ${port}...${NC}"
        kill $pid 2>/dev/null
        sleep 1
        # Force kill if still running
        if kill -0 $pid 2>/dev/null; then
            echo -e "${YELLOW}Force stopping ${name}...${NC}"
            kill -9 $pid 2>/dev/null
        fi
        echo -e "${GREEN}${name} stopped${NC}"
    else
        echo -e "${YELLOW}${name} not running on port ${port}${NC}"
    fi
}

# Stop agents by matching python process
echo ""
echo "Stopping agents by process..."
pkill -f "python.*agents/weather_agent/main.py" 2>/dev/null
pkill -f "python.*agents/search_agent/main.py" 2>/dev/null
pkill -f "python.*agents/location_agent/main.py" 2>/dev/null
pkill -f "python.*agents/coordinator_agent/main.py" 2>/dev/null

sleep 1

# Verify by checking ports
echo ""
echo "Verifying agents are stopped..."
echo ""

stop_agent_by_port "Coordinator Agent" 8000
stop_agent_by_port "Weather Agent" 8001
stop_agent_by_port "Search Agent" 8002
stop_agent_by_port "Location Agent" 8003

echo ""
echo -e "${GREEN}All agents stopped!${NC}"
