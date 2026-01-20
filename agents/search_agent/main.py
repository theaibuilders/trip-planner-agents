"""Search Agent - Uses Bright Data SERP API for web search."""
import os
import httpx
from urllib.parse import quote_plus
from dotenv import load_dotenv
from agentfield import Agent, AIConfig

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.models import SearchResult, SearchResults

load_dotenv()

# Initialize the Agentfield agent
app = Agent(
    node_id="search-agent",
    ai_config=AIConfig(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0.2
    ),
    enable_did=False
)

BRIGHT_DATA_API_URL = "https://api.brightdata.com/request"


@app.skill()
async def search_web(query: str, num_results: int = 10) -> dict:
    """
    Search the web using Bright Data SERP API.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default 10)
    
    Returns:
        Search results with titles, URLs, and snippets
    """
    api_key = os.getenv("BRIGHT_DATA_API_KEY")
    zone = os.getenv("BRIGHT_DATA_ZONE", "serp_api1")
    
    if not api_key:
        return SearchResults(
            query=query,
            results=[SearchResult(
                title="API Key Missing",
                url="",
                snippet="BRIGHT_DATA_API_KEY environment variable not set"
            )]
        ).model_dump()
    
    encoded_query = quote_plus(query)
    # Add brd_json=1 to get parsed JSON response
    search_url = f"https://www.google.com/search?q={encoded_query}&num={num_results}&brd_json=1"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "zone": zone,
        "url": search_url,
        "format": "raw"
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                BRIGHT_DATA_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        # Handle different response formats
        organic_results = data.get("organic", []) or data.get("results", []) or []
        
        for item in organic_results[:num_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", "") or item.get("url", ""),
                snippet=item.get("snippet", "") or item.get("description", "")
            ))
        
        if not results:
            return SearchResults(
                query=query,
                results=[SearchResult(
                    title="No Results",
                    url="",
                    snippet="No search results found. Check API configuration."
                )]
            ).model_dump()
        
        return SearchResults(query=query, results=results).model_dump()
    
    except httpx.HTTPStatusError as e:
        error_detail = ""
        try:
            error_detail = e.response.text[:200]
        except:
            pass
        return SearchResults(
            query=query,
            results=[SearchResult(
                title="API Error",
                url="",
                snippet=f"HTTP {e.response.status_code}: {error_detail}"
            )]
        ).model_dump()
    except Exception as e:
        return SearchResults(
            query=query,
            results=[SearchResult(
                title="Error",
                url="",
                snippet=str(e)
            )]
        ).model_dump()


@app.skill()
async def search_attractions(destination: str, num_results: int = 5) -> dict:
    """
    Search for tourist attractions at a destination.
    
    Args:
        destination: Name of the destination
        num_results: Number of results to return
    
    Returns:
        Search results for attractions
    """
    query = f"best tourist attractions things to do in {destination}"
    return await search_web(query, num_results)


@app.skill()
async def search_restaurants(destination: str, cuisine: str = "", num_results: int = 5) -> dict:
    """
    Search for restaurants at a destination.
    
    Args:
        destination: Name of the destination
        cuisine: Type of cuisine (optional)
        num_results: Number of results to return
    
    Returns:
        Search results for restaurants
    """
    cuisine_part = f"{cuisine} " if cuisine else ""
    query = f"best {cuisine_part}restaurants in {destination}"
    return await search_web(query, num_results)


@app.skill()
async def search_accommodations(destination: str, budget: str = "", num_results: int = 5) -> dict:
    """
    Search for accommodations at a destination.
    
    Args:
        destination: Name of the destination
        budget: Budget level (luxury, mid-range, budget)
        num_results: Number of results to return
    
    Returns:
        Search results for accommodations
    """
    budget_part = f"{budget} " if budget else ""
    query = f"best {budget_part}hotels accommodations in {destination}"
    return await search_web(query, num_results)


@app.skill()
async def search_travel_tips(destination: str, num_results: int = 5) -> dict:
    """
    Search for travel tips for a destination.
    
    Args:
        destination: Name of the destination
        num_results: Number of results to return
    
    Returns:
        Search results for travel tips
    """
    query = f"travel tips advice visiting {destination}"
    return await search_web(query, num_results)


@app.skill()
async def search_events(
    destination: str,
    event_type: str = "",
    date_range: str = "",
    num_results: int = 10
) -> dict:
    """
    Search for events across platforms like lu.ma, meetup.com, eventbrite, etc.
    
    Args:
        destination: Location/city to search events in
        event_type: Type of event (e.g., "tech", "ai", "music", "food", "networking")
        date_range: Date range for events (e.g., "this week", "January 2026")
        num_results: Number of results to return
    
    Returns:
        Search results for events from various platforms
    """
    # Build search query targeting event platforms
    event_platforms = "site:lu.ma OR site:meetup.com OR site:eventbrite.com OR site:events.com"
    
    type_part = f"{event_type} " if event_type else ""
    date_part = f" {date_range}" if date_range else ""
    
    query = f"{type_part}events in {destination}{date_part} ({event_platforms})"
    
    return await search_web(query, num_results)


@app.skill()
async def search_tech_events(
    destination: str,
    topic: str = "",
    date_range: str = "",
    num_results: int = 10
) -> dict:
    """
    Search specifically for tech, AI, and startup events.
    
    Args:
        destination: Location/city to search events in
        topic: Specific tech topic (e.g., "AI", "machine learning", "web3", "startup")
        date_range: Date range for events (e.g., "this week", "January 2026")
        num_results: Number of results to return
    
    Returns:
        Search results for tech events
    """
    # Target tech event platforms and communities
    event_platforms = "site:lu.ma OR site:meetup.com OR site:eventbrite.com OR site:techmeme.com/events"
    
    topic_part = f"{topic} " if topic else "tech AI startup "
    date_part = f" {date_range}" if date_range else ""
    
    query = f"{topic_part}events meetup conference in {destination}{date_part} ({event_platforms})"
    
    return await search_web(query, num_results)


@app.reasoner()
async def summarize_search_results(
    search_results: dict,
    context: str = ""
) -> dict:
    """
    Summarize and extract key information from search results.
    
    Args:
        search_results: Search results from search_web
        context: Additional context for summarization
    
    Returns:
        Summarized information
    """
    context_str = f"Context: {context}" if context else ""
    
    result = await app.ai(
        system="""You are a travel research assistant. 
        Analyze search results and extract the most useful information 
        for trip planning. Focus on practical, actionable insights.""",
        user=f"""Search results: {search_results}
        
{context_str}

Extract and summarize:
1. Top recommendations with brief descriptions
2. Key practical information (hours, prices, locations)
3. Insider tips or important notes
4. Any warnings or things to avoid""",
        schema={
            "type": "object",
            "properties": {
                "top_recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "source_url": {"type": "string"}
                        }
                    }
                },
                "practical_info": {"type": "array", "items": {"type": "string"}},
                "insider_tips": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["top_recommendations"]
        }
    )
    
    return result


if __name__ == "__main__":
    app.run(port=8002)
