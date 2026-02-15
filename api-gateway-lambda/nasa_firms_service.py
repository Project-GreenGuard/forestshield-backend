import requests
import os
import csv
from io import StringIO

def get_nasa_fires():
    """
    Fetch wildfire data from NASA FIRMS
    Returns a list of fire dictionaries with lat/lng coordinates
    """
    NASA_API_KEY = os.getenv("NASA_MAP_KEY")
    
    # Check if API key is set
    if not NASA_API_KEY:
        print("⚠️ Warning: NASA_MAP_KEY not set. Returning empty fire data.")
        return []
    
    # Ontario bounding box
    # west, south, east, north
    bbox = "-95.5,41.5,-74.0,56.9"

    # Use CSV format as JSON endpoint returns HTML error pages
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{NASA_API_KEY}/VIIRS_SNPP_NRT/{bbox}/1"

    try:
        print("🔥 Calling NASA FIRMS API...")
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse CSV response
        csv_data = response.text.strip()
        
        if not csv_data or csv_data.startswith('<!DOCTYPE'):
            print("⚠️ Warning: NASA API returned HTML instead of data")
            return []
        
        # Parse CSV into list of dictionaries
        csv_reader = csv.DictReader(StringIO(csv_data))
        fires = []
        
        for row in csv_reader:
            # Convert CSV row to dictionary with standardized keys
            fire = {
                'latitude': float(row.get('latitude', 0)),
                'longitude': float(row.get('longitude', 0)),
                'brightness': float(row.get('bright_ti4', 0)),
                'acq_date': row.get('acq_date', ''),
                'acq_time': row.get('acq_time', ''),
                'satellite': row.get('satellite', ''),
                'confidence': row.get('confidence', ''),
                'frp': float(row.get('frp', 0)) if row.get('frp') else 0.0
            }
            fires.append(fire)

        print(f"🔥 NASA returned {len(fires)} fires")
        return fires
    
    except requests.exceptions.RequestException as e:
        print(f"NASA API error: {e}")
        return []
    except Exception as e:
        print(f"NASA error: {e}")
        import traceback
        traceback.print_exc()
        return []
