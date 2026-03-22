import csv
import os
from io import StringIO

import requests


def get_nasa_fires():
    """
    Fetch wildfire hotspots from NASA FIRMS (VIIRS SNPP NRT) for the Ontario bbox.
    Requires NASA_MAP_KEY; returns [] if unset or on error.
    """
    api_key = os.getenv("NASA_MAP_KEY", "").strip()
    if not api_key:
        print("Warning: NASA_MAP_KEY not set. Returning empty fire data.")
        return []

    bbox = "-95.5,41.5,-74.0,56.9"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/VIIRS_SNPP_NRT/{bbox}/1"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        text = response.text.strip()
        if not text or text.startswith("<!DOCTYPE"):
            print("Warning: NASA API returned non-CSV body")
            return []

        reader = csv.DictReader(StringIO(text))
        fires = []
        for row in reader:
            fires.append(
                {
                    "latitude": float(row.get("latitude", 0)),
                    "longitude": float(row.get("longitude", 0)),
                    "brightness": float(row.get("bright_ti4", 0)),
                    "acq_date": row.get("acq_date", ""),
                    "acq_time": row.get("acq_time", ""),
                    "satellite": row.get("satellite", ""),
                    "confidence": row.get("confidence", ""),
                    "frp": float(row.get("frp", 0)) if row.get("frp") else 0.0,
                }
            )
        print(f"NASA FIRMS returned {len(fires)} fires")
        return fires
    except requests.exceptions.RequestException as e:
        print(f"NASA API error: {e}")
        return []
    except Exception as e:
        print(f"NASA error: {e}")
        import traceback

        traceback.print_exc()
        return []
