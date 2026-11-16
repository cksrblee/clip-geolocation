import unittest

import numpy as np

from src.interventions.scrambling import patch_mixed_across_regions, patch_shuffle


class PatchShuffleTests(unittest.TestCase):
    def test_shuffle_is_deterministic_and_preserves_grid_pixels(self):
        image = np.arange(8 * 8 * 3, dtype=np.uint16).reshape(8, 8, 3)
        first = patch_shuffle(image, patch_size=4, rng=np.random.default_rng(7))
        second = patch_shuffle(image, patch_size=4, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(
            np.sort(first.reshape(-1)), np.sort(image.reshape(-1))
        )

    def test_invalid_patch_size(self):
        with self.assertRaises(ValueError):
            patch_shuffle(np.zeros((4, 4, 3), dtype=np.uint8), patch_size=0)

    def test_remainder_modes_and_patch_mixed(self):
        image = np.ones((10, 10, 3), dtype=np.uint8) * 10
        partner = np.ones((10, 10, 3), dtype=np.uint8) * 20
        zeroed = patch_shuffle(
            image, patch_size=4, rng=np.random.default_rng(1), remainder="zero"
        )
        preserved = patch_shuffle(
            image, patch_size=4, rng=np.random.default_rng(1), remainder="preserve"
        )
        cropped = patch_shuffle(
            image, patch_size=4, rng=np.random.default_rng(1), remainder="crop"
        )
        mixed = patch_mixed_across_regions(image, partner, patch_size=4)
        self.assertEqual(zeroed.shape, image.shape)
        self.assertEqual(cropped.shape, (8, 8, 3))
        self.assertEqual(int(zeroed[8:, :, :].sum()), 0)
        np.testing.assert_array_equal(preserved[8:, :, :], image[8:, :, :])
        np.testing.assert_array_equal(mixed[0:4, 0:4, :], 10)
        np.testing.assert_array_equal(mixed[0:4, 4:8, :], 20)


if __name__ == "__main__":
    unittest.main()
