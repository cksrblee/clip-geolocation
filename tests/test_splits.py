import unittest
from pathlib import Path

import pandas as pd

from src.data.build_splits import (
    area_normalized_sample,
    load_regions,
    region_stratified_split,
    spatial_overlap_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class DatasetSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = pd.read_csv(REPO_ROOT / "dataset" / "metadata.csv")
        cls.regions = load_regions(REPO_ROOT / "configs" / "la_county_regions.json")

    def test_area_normalized_sample_matches_table_i_population(self):
        sampled, summary = area_normalized_sample(self.metadata, self.regions, seed=42)
        self.assertEqual(len(sampled), 9085)
        self.assertAlmostEqual(summary["target_density"].iloc[0], 194.493404, places=5)
        self.assertEqual(
            set(sampled["region_id"]), {region["id"] for region in self.regions}
        )

    def test_split_is_deterministic_stratified_and_disjoint(self):
        sampled, _ = area_normalized_sample(self.metadata, self.regions, seed=42)
        first = region_stratified_split(sampled, seed=42)
        second = region_stratified_split(sampled, seed=42)
        self.assertEqual(
            {name: len(frame) for name, frame in first.items()},
            {
                "train": 6359,
                "val": 1363,
                "test": 1363,
            },
        )
        for name in first:
            self.assertTrue(first[name]["filename"].equals(second[name]["filename"]))
            self.assertEqual(
                set(first[name]["region_id"]), {region["id"] for region in self.regions}
            )
        filename_sets = [set(frame["filename"]) for frame in first.values()]
        self.assertFalse(filename_sets[0] & filename_sets[1])
        self.assertFalse(filename_sets[0] & filename_sets[2])
        self.assertFalse(filename_sets[1] & filename_sets[2])

    def test_spatial_overlap_matches_reported_split_audit(self):
        train = pd.read_csv(REPO_ROOT / "dataset" / "train_metadata.csv")
        test = pd.read_csv(REPO_ROOT / "dataset" / "test_metadata.csv")
        audit = spatial_overlap_audit(train, test)
        self.assertEqual(int(audit["exact_train_coordinate"].sum()), 755)
        self.assertEqual(int((audit["nearest_train_distance_m"] <= 50).sum()), 1086)
        self.assertEqual(int((audit["nearest_train_distance_m"] > 50).sum()), 277)


if __name__ == "__main__":
    unittest.main()
