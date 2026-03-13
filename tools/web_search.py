import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def web_search(query: str) -> str:
    """
    Searches the web for real-time information using Tavily API.
    Tavily is specifically designed for AI agents.
    """
    if not TAVILY_API_KEY:
        return json.dumps({"error": "Tavily API key is missing in .env file."})

    tavily_url = "https://api.tavily.com/search"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Tavily API Payload
    data = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic", # or "advanced" for deeper search
        "max_results": 3,
        "include_answer": False
    }
    
    try:
        response = requests.post(tavily_url, headers=headers, json=data, timeout=10)
        
        if response.status_code != 200:
            return json.dumps({
                "error": f"Tavily API error (Status {response.status_code})",
                "details": response.json().get("detail", "Unknown error")
            })
            
        search_data = response.json()
        results = search_data.get("results", [])
        
        if not results:
            return json.dumps({"error": f"No Tavily search results found for query: '{query}'"})
            
        simplified_results = []
        for item in results:
            simplified_results.append({
                "title": item.get("title"),
                "snippet": item.get("content"), # Tavily calls snippets 'content'
                "source": item.get("url")
            })
            
        return json.dumps(simplified_results)

    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Error connecting to Tavily API: {str(e)}"})
