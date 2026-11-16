import unittest

from src.evaluation.metrics import (
    accuracy_retention,
    calculate_metrics,
    calculate_shortcut_reliance,
    calculate_structure_sufficiency,
    prediction_switch_rate,
    top1_accuracy,
)


class ProbingMetricTests(unittest.TestCase):
    def test_shortcut_reliance_is_accuracy_drop(self):
        self.assertAlmostEqual(calculate_shortcut_reliance(80.85, 73.22), 7.63, places=2)

    def test_accuracy_and_structure_sufficiency(self):
        self.assertEqual(top1_accuracy([0, 1, 1, 0], [0, 1, 0, 0]), 75.0)
        self.assertEqual(calculate_structure_sufficiency(30.0, 40.0), 0.75)
        self.assertEqual(accuracy_retention(30.0, 40.0), 0.75)

    def test_switch_rate_is_sample_level_prediction_change(self):
        self.assertEqual(prediction_switch_rate([0, 1, 2, 3], [0, 2, 2, 0]), 50.0)
        with self.assertRaises(ValueError):
            prediction_switch_rate([0, 1], [0])

    def test_centroid_error_matches_original_region_centroid_formula(self):
        # Correct predictions incur 0 km error; mispredictions incur inter-centroid distance.
        metrics = calculate_metrics(
            predictions=[0, 1],
            ground_truth=[0, 0],
            id_to_region={0: "a", 1: "b"},
            region_centers={"a": (0.0, 0.0), "b": (0.0, 1.0)},
        )
        self.assertEqual(metrics["accuracy"], 50.0)
        self.assertEqual(metrics["raw_errors"][0], 0.0)
        self.assertAlmostEqual(metrics["mean_error_km"], 111.195 / 2.0, places=1)


if __name__ == "__main__":
    unittest.main()
