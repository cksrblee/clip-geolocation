import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.download_data import save_metadata_csv
from src.data.downloader import StreetViewDownloader


class DownloaderBugfixTests(unittest.TestCase):
    def test_download_region_collects_exact_location_count_and_skips_http_errors(self):
        bbox = [
            {"latitude": 34.0, "longitude": -118.3},
            {"latitude": 34.1, "longitude": -118.2},
        ]
        call_counter = {"meta": 0}

        def fake_check(lat, lon, radius=50):
            call_counter["meta"] += 1
            idx = call_counter["meta"]
            return {
                "status": "OK",
                "available": True,
                "actual_lat": 34.05 + idx * 0.001,
                "actual_lon": -118.25,
                "distance": 5.0,
                "date": "2026-01",
                "pano_id": f"pano_{idx}",
            }

        def fake_get(url, params=None, timeout=15):
            resp = MagicMock()
            # Simulate heading 270 failing on the first location only
            if params["heading"] == 270 and call_counter["meta"] == 1:
                resp.status_code = 404
                resp.content = b""
            else:
                resp.status_code = 200
                resp.content = b"fake_jpeg_bytes"
            return resp

        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = StreetViewDownloader(api_key="dummy", output_dir=tmpdir)
            with (
                patch.object(downloader.validator, "check_availability", side_effect=fake_check),
                patch("src.data.downloader.requests.get", side_effect=fake_get),
            ):
                records = downloader.download_region(
                    region_id="downtown_la",
                    bbox_corners=bbox,
                    num_locations=2,
                    headings=[0, 90, 180, 270],
                    prefix="dtla",
                )

            # Location 0 had 3 successful headings (270 failed), location 1 had 4 -> 7 records across 2 locations
            self.assertEqual(len(records), 7)
            self.assertEqual(len({r["pano_id"] for r in records}), 2)
            self.assertEqual(records[0]["filename"], "streetview_downtown_la_dtla_0000_h000.jpg")
            self.assertEqual(records[3]["filename"], "streetview_downtown_la_dtla_0001_h000.jpg")

            csv_path = Path(tmpdir) / "metadata.csv"
            save_metadata_csv(records, str(csv_path))
            self.assertTrue(csv_path.is_file())


if __name__ == "__main__":
    unittest.main()
