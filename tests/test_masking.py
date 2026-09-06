import unittest

import numpy as np

from src.interventions.masking import random_rectangle_control


class MatchedMaskTests(unittest.TestCase):
    def test_random_control_matches_requested_area_exactly(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        _, mask = random_rectangle_control(
            image, target_area=137, rng=np.random.default_rng(42)
        )
        self.assertEqual(int(mask.sum()), 137)


if __name__ == "__main__":
    unittest.main()
