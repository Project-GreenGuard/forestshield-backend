import csv
import os
from io import StringIO

import requests


def get_nasa_fires():
    """
    Fetch wildfire hotspots from NASA FIRMS (VIIRS SNPP NRT) for the Ontario/GTA bbox.
    Requires NASA_MAP_KEY; returns [] if unset or on error.
    Applies basic filtering to reduce low-quality thermal anomalies.
    """
    api_key = os.getenv("NASA_MAP_KEY", "").strip()
    if not api_key:
        print("Warning: NASA_MAP_KEY not set. Returning empty fire data.")
        return []

    bbox = "-83.5,41.5,-74.5,45.5"
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

        total_rows = 0
        skipped_confidence = 0
        skipped_frp = 0
        skipped_coords = 0

        for row in reader:
            total_rows += 1

            try:
                lat = float(row.get("latitude", 0) or 0)
                lng = float(row.get("longitude", 0) or 0)
                brightness = float(row.get("bright_ti4", 0) or 0)
                frp = float(row.get("frp", 0) or 0)
                confidence = (row.get("confidence", "") or "").strip().lower()

                if lat == 0 or lng == 0:
                    skipped_coords += 1
                    continue

                # Keep only high-confidence detections
                if confidence != "h":
                    skipped_confidence += 1
                    continue

                # Remove very weak heat sources
                if frp < 5:
                    skipped_frp += 1
                    continue

                fires.append(
                    {
                        "latitude": lat,
                        "longitude": lng,
                        "brightness": brightness,
                        "acq_date": row.get("acq_date", ""),
                        "acq_time": row.get("acq_time", ""),
                        "satellite": row.get("satellite", ""),
                        "confidence": confidence,
                        "frp": frp,
                    }
                )

            except (ValueError, TypeError):
                continue

        print(f"NASA FIRMS total rows: {total_rows}")
        print(f"Skipped low confidence: {skipped_confidence}")
        print(f"Skipped low FRP: {skipped_frp}")
        print(f"Skipped invalid coords: {skipped_coords}")
        print(f"NASA FIRMS returned {len(fires)} filtered fires")

        return fires

    except requests.exceptions.RequestException as e:
        print(f"NASA API error: {e}")
        return []
    except Exception as e:
        print(f"NASA error: {e}")
        import traceback
        traceback.print_exc()
        return []