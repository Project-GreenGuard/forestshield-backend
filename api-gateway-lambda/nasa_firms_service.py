import requests
import os

NASA_API_KEY = os.getenv("NASA_MAP_KEY")  # put in .env

def get_nasa_fires():
    """
    Fetch wildfire data from NASA FIRMS
    """
    # Canada bounding box
    # west, south, east, north
    bbox = "-141,41,-52,83"

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/json/{NASA_API_KEY}/VIIRS_SNPP_NRT/{bbox}/1"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return data
    except Exception as e:
        print("NASA error:", e)
        return []
