#!/usr/bin/env python3
"""
download_data.py: Download Google Street View images and metadata for defined geographic regions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys


# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.downloader import StreetViewDownloader


FIELDNAMES = [
    "filename",
    "region_id",
    "latitude",
    "longitude",
    "heading",
    "pitch",
    "fov",
    "pano_id",
    "date",
]


def save_metadata_csv(records: list[dict], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in FIELDNAMES})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Street View images for Geolocation dataset."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("GOOGLE_MAPS_API_KEY", ""),
        help="Google Maps API Key (or set GOOGLE_MAPS_API_KEY env var)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/la_county_regions.json",
        help="Path to region configuration JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Output directory for images and metadata",
    )
    parser.add_argument(
        "--locations_per_region",
        type=int,
        default=100,
        help="Number of distinct valid locations per region",
    )
    parser.add_argument(
        "--headings",
        type=int,
        nargs="+",
        default=[0, 90, 180, 270],
        help="Camera headings to download per location",
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "Error: Google Maps API key is required. Pass --api_key or set GOOGLE_MAPS_API_KEY."
        )
        sys.exit(1)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = (
        os.path.join(repo_root, args.config)
        if not os.path.isabs(args.config)
        else args.config
    )
    output_dir = (
        os.path.join(repo_root, args.output_dir)
        if not os.path.isabs(args.output_dir)
        else args.output_dir
    )

    with open(config_path, "r", encoding="utf-8") as handle:
        config_data = json.load(handle)
    regions_data = config_data["regions"]
    defaults = config_data.get("default_settings", {})

    downloader = StreetViewDownloader(api_key=args.api_key, output_dir=output_dir)

    all_records: list[dict] = []
    for region in regions_data:
        region_id = region["id"]
        corners = region["bbox"]["corners"]
        prefix = region.get("output", {}).get("prefix")
        radius = int(region.get("validation", {}).get("radius", 50))
        records = downloader.download_region(
            region_id=region_id,
            bbox_corners=corners,
            num_locations=args.locations_per_region,
            headings=args.headings,
            size=str(defaults.get("image_size", "640x640")),
            fov=int(defaults.get("fov", 90)),
            pitch=int(defaults.get("pitch", 0)),
            prefix=prefix,
            radius=radius,
            delay_seconds=float(defaults.get("delay_seconds", 0.0)),
        )
        all_records.extend(records)

    metadata_csv = os.path.join(output_dir, "metadata.csv")
    save_metadata_csv(all_records, metadata_csv)
    print(
        f"Data collection completed! Saved {len(all_records)} image records to {metadata_csv}"
    )


if __name__ == "__main__":
    main()
