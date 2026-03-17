import os
from langchain_google_community import GoogleSearchAPIWrapper
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def google_search(query: str) -> str:
    """
    Searches the web using Google Custom Search API.
    Returns detailed results including snippets and source links.
    """
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_CSE_ID"):
        return json.dumps({"error": "Google API Key or Search Engine ID is missing."})

    try:
        # 建立 Google 搜尋包裝器
        # k=5 代表回傳 5 筆結果
        search = GoogleSearchAPIWrapper(k=5)
        
        # 獲取詳細結果 (包含 title, link, snippet)
        results = search.results(query, num_results=5)
        
        if not results:
            return json.dumps({"error": f"No Google results found for: '{query}'"})
            
        return json.dumps(results)

    except Exception as e:
        return json.dumps({"error": f"Google Search failed: {str(e)}"})
