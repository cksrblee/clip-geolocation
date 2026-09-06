import unittest

from src.evaluation.metrics import (
    accuracy_retention,
    calculate_metrics,
    prediction_switch_rate,
    top1_accuracy,
)


class ProbingMetricTests(unittest.TestCase):
    def test_switch_rate_is_sample_level_prediction_change(self):
        self.assertEqual(prediction_switch_rate([0, 1, 2, 3], [0, 2, 2, 0]), 50.0)

    def test_accuracy_and_retention(self):
        self.assertEqual(top1_accuracy([0, 1, 1, 0], [0, 1, 0, 0]), 75.0)
        self.assertEqual(accuracy_retention(30.0, 40.0), 0.75)

    def test_switch_rate_rejects_unaligned_samples(self):
        with self.assertRaises(ValueError):
            prediction_switch_rate([0, 1], [0])

    def test_centroid_error_uses_image_coordinate(self):
        metrics = calculate_metrics(
            predictions=[0],
            ground_truth=[0],
            sample_coordinates=[(0.0, 1.0)],
            id_to_region={0: "a"},
            region_centers={"a": (0.0, 0.0)},
        )
        self.assertAlmostEqual(metrics["mean_error_km"], 111.195, places=2)


if __name__ == "__main__":
    unittest.main()
