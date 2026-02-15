import requests
import os

NASA_API_KEY = os.getenv("NASA_MAP_KEY")  # put in .env

def get_nasa_fires():
    """
    Fetch wildfire data from NASA FIRMS
    """
    # Ontario bounding box
    # west, south, east, north
    bbox = "-95.5,41.5,-74.0,56.9"
    days= 5

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/json/{NASA_API_KEY}/VIIRS_SNPP_NRT/{bbox}/1"

    try:
        print("🔥 Calling NASA FIRMS API...")
        response = requests.get(url, timeout=15)
        data = response.json()

        print(f"🔥 NASA returned {len(data)} fires")
        return data
    
    except Exception as e:
        print("NASA error:", e)
        return []
