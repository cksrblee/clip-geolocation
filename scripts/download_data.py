#!/usr/bin/env python3
"""
download_data.py: Download Google Street View images and metadata for defined geographic regions.
"""

import os
import sys
import argparse
import json

# Ensure repository root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.downloader import StreetViewDownloader


def main():
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

    with open(config_path, "r") as f:
        regions_data = json.load(f)["regions"]

    downloader = StreetViewDownloader(api_key=args.api_key, output_dir=output_dir)

    all_records = []
    for region in regions_data:
        region_id = region["id"]
        corners = region["bbox"]["corners"]
        records = downloader.download_region(
            region_id=region_id,
            bbox_corners=corners,
            num_locations=args.locations_per_region,
            headings=args.headings,
        )
        all_records.extend(records)

    print(f"Data collection completed! Total images: {len(all_records)}")


if __name__ == "__main__":
    main()
