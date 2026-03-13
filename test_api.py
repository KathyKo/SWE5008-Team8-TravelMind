import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CUSTOM_SEARCH_API_KEY")
CX = os.getenv("CUSTOM_SEARCH_CX")

print(f"DEBUG: API_KEY='{API_KEY}'")
print(f"DEBUG: CX='{CX}'")

if not API_KEY or not CX:
    print("ERROR: Missing API_KEY or CX in .env file.")
else:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CX,
        "q": "test",
        "num": 1
    }
    try:
        response = requests.get(url, params=params)
        print(f"HTTP Status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response Body: {response.text}")
        else:
            print("Success! The API is working correctly.")
    except Exception as e:
        print(f"Error: {e}")
