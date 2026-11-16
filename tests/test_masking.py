import unittest

import numpy as np

from src.interventions.masking import random_rectangle_control


class MatchedMaskTests(unittest.TestCase):
    def test_random_square_control_uses_side_sqrt_area(self):
        image = np.zeros((40, 50, 3), dtype=np.uint8)
        _, mask = random_rectangle_control(
            image, target_area=100, rng=np.random.default_rng(42)
        )
        self.assertEqual(int(mask.sum()), 100)

    def test_zero_area_falls_back_to_five_percent(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, mask = random_rectangle_control(
            image, target_area=0, rng=np.random.default_rng(42)
        )
        # int(sqrt(500)) = 22 -> 22 * 22 = 484
        self.assertEqual(int(mask.sum()), 484)


if __name__ == "__main__":
    unittest.main()
