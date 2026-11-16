"""
Google Street View Image & Metadata Downloader.
Provides metadata verification, spatial sampling within regional bounding boxes,
and error handling.
"""

from __future__ import annotations

import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence

import requests


class StreetViewValidator:
    """Validator using Google Maps Street View Metadata API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"

    def check_availability(
        self, lat: float, lon: float, radius: int = 50
    ) -> Dict[str, Any]:
        """Check if a Street View panorama is available at the given coordinate."""
        params = {"location": f"{lat},{lon}", "radius": radius, "key": self.api_key}

        try:
            response = requests.get(self.metadata_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "OK" and "location" in data:
                    actual_lat = data["location"]["lat"]
                    actual_lon = data["location"]["lng"]
                    distance = self._haversine(lat, lon, actual_lat, actual_lon)
                    return {
                        "status": "OK",
                        "available": True,
                        "actual_lat": actual_lat,
                        "actual_lon": actual_lon,
                        "distance": distance,
                        "date": data.get("date", "unknown"),
                        "pano_id": data.get("pano_id", "unknown"),
                    }
                return {"status": data.get("status", "UNKNOWN"), "available": False}
            return {"status": f"HTTP_{response.status_code}", "available": False}
        except Exception as exc:
            return {"status": f"ERROR_{exc}", "available": False}

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance in meters."""
        earth_radius_m = 6371000.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return earth_radius_m * c


class StreetViewDownloader:
    """Street View image downloader supporting multiple headings and metadata logging."""

    def __init__(self, api_key: str, output_dir: str = "dataset"):
        self.api_key = api_key
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        self.validator = StreetViewValidator(api_key)
        self.base_url = "https://maps.googleapis.com/maps/api/streetview"

    def download_region(
        self,
        region_id: str,
        bbox_corners: List[Dict[str, float]],
        num_locations: int = 100,
        headings: Optional[Sequence[int]] = None,
        size: str = "640x640",
        fov: int = 90,
        pitch: int = 0,
        prefix: Optional[str] = None,
        radius: int = 50,
        delay_seconds: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Sample valid coordinates within bounding box and download Street View images."""
        if headings is None:
            headings = (0, 90, 180, 270)
        heading_list = [int(h) for h in headings]

        lats = [c["latitude"] for c in bbox_corners]
        lons = [c["longitude"] for c in bbox_corners]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        collected: List[Dict[str, Any]] = []
        sampled_panos: set[str] = set()
        locations_completed = 0
        attempts = 0
        max_attempts = max(num_locations * 15, 1)

        print(f"[{region_id}] Starting collection of {num_locations} locations...")

        while locations_completed < num_locations and attempts < max_attempts:
            attempts += 1
            rand_lat = random.uniform(min_lat, max_lat)
            rand_lon = random.uniform(min_lon, max_lon)

            val_res = self.validator.check_availability(
                rand_lat, rand_lon, radius=radius
            )
            if not val_res.get("available"):
                continue

            pano_id = str(val_res.get("pano_id", ""))
            dedup_key = pano_id or f"{val_res['actual_lat']:.6f},{val_res['actual_lon']:.6f}"
            if dedup_key in sampled_panos:
                continue

            sampled_panos.add(dedup_key)
            loc_idx = locations_completed
            actual_lat = val_res["actual_lat"]
            actual_lon = val_res["actual_lon"]
            date_str = val_res.get("date", "unknown")

            location_records: List[Dict[str, Any]] = []
            for h in heading_list:
                stem_prefix = f"{region_id}_{prefix}" if prefix else region_id
                filename = f"streetview_{stem_prefix}_{loc_idx:04d}_h{h:03d}.jpg"
                filepath = os.path.join(self.image_dir, filename)

                if not os.path.exists(filepath):
                    params = {
                        "size": size,
                        "location": f"{actual_lat},{actual_lon}",
                        "heading": h,
                        "pitch": pitch,
                        "fov": fov,
                        "key": self.api_key,
                    }
                    try:
                        resp = requests.get(self.base_url, params=params, timeout=15)
                        if resp.status_code != 200 or not resp.content:
                            print(
                                f"Warning: HTTP {resp.status_code} for {filename}; skipping."
                            )
                            continue
                        with open(filepath, "wb") as img_f:
                            img_f.write(resp.content)
                    except Exception as exc:
                        print(f"Error downloading {filename}: {exc}")
                        continue
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)

                location_records.append(
                    {
                        "filename": filename,
                        "region_id": region_id,
                        "latitude": actual_lat,
                        "longitude": actual_lon,
                        "heading": h,
                        "pitch": pitch,
                        "fov": fov,
                        "pano_id": pano_id,
                        "date": date_str,
                    }
                )

            if location_records:
                locations_completed += 1
                collected.extend(location_records)
                if locations_completed % 10 == 0 or locations_completed == num_locations:
                    print(
                        f"[{region_id}] Progress: {locations_completed}/{num_locations} locations "
                        f"({len(collected)} images)."
                    )

        return collected
