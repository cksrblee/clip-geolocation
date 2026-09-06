"""
Google Street View Image & Metadata Downloader
Provides robust metadata verification, spatial sampling within regional bounding boxes,
and error handling.
"""

import json
import requests
import csv
import os
import math
import random
import time
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime


class StreetViewValidator:
    """Validator using Google Maps Street View Metadata API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"

    def check_availability(
        self, lat: float, lon: float, radius: int = 50
    ) -> Dict[str, Any]:
        """Check if Street View panorama is available at given coordinate.

        Args:
            lat: Latitude
            lon: Longitude
            radius: Search radius in meters (default 50)

        Returns:
            Dictionary containing availability status, actual coordinates, pano_id, and capture date.
        """
        params = {"location": f"{lat},{lon}", "radius": radius, "key": self.api_key}

        try:
            response = requests.get(self.metadata_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "OK":
                    actual_lat = data["location"]["lat"]
                    actual_lon = data["location"]["lng"]

                    # Distance between queried and returned panorama
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
                else:
                    return {"status": data.get("status", "UNKNOWN"), "available": False}
            else:
                return {"status": f"HTTP_{response.status_code}", "available": False}
        except Exception as e:
            return {"status": f"ERROR_{str(e)}", "available": False}

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance in meters."""
        R = 6371000  # Earth radius in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c


class StreetViewDownloader:
    """Street View Image Downloader supporting headings, pitches, and metadata recording."""

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
        headings: List[int] = [0, 90, 180, 270],
        size: str = "640x640",
        fov: int = 90,
        pitch: int = 0,
    ) -> List[Dict[str, Any]]:
        """Sample valid coordinates within bounding box and download Street View images.

        Args:
            region_id: Unique string identifier for region
            bbox_corners: List of 4 corner dicts {'latitude': ..., 'longitude': ...}
            num_locations: Target number of unique valid locations
            headings: List of camera heading angles (0=North, 90=East, etc.)
            size: Image dimensions (e.g. '640x640')
            fov: Field of view (default 90)
            pitch: Camera pitch angle (default 0)
        """
        lats = [c["latitude"] for c in bbox_corners]
        lons = [c["longitude"] for c in bbox_corners]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        collected = []
        sampled_panos = set()
        attempts = 0
        max_attempts = num_locations * 15

        print(f"[{region_id}] Starting collection of {num_locations} locations...")

        while len(collected) < num_locations and attempts < max_attempts:
            attempts += 1
            rand_lat = random.uniform(min_lat, max_lat)
            rand_lon = random.uniform(min_lon, max_lon)

            val_res = self.validator.check_availability(rand_lat, rand_lon)
            if not val_res.get("available"):
                continue

            pano_id = val_res.get("pano_id")
            if pano_id in sampled_panos:
                continue

            sampled_panos.add(pano_id)
            loc_idx = len(collected) + 1
            actual_lat = val_res["actual_lat"]
            actual_lon = val_res["actual_lon"]
            date_str = val_res.get("date", "unknown")

            for h in headings:
                filename = f"streetview_{region_id}_{loc_idx:04d}_h{h}.jpg"
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
                        if resp.status_code == 200:
                            with open(filepath, "wb") as img_f:
                                img_f.write(resp.content)
                    except Exception as e:
                        print(f"Error downloading {filename}: {e}")
                        continue

                collected.append(
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

            if len(collected) % 50 == 0:
                print(
                    f"[{region_id}] Progress: {len(collected)} / {num_locations * len(headings)} images downloaded."
                )

        return collected
