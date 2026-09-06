import unittest

import numpy as np

from src.interventions.scrambling import patch_shuffle


class PatchShuffleTests(unittest.TestCase):
    def test_shuffle_is_deterministic_and_preserves_pixels(self):
        image = np.arange(10 * 10 * 3, dtype=np.uint16).reshape(10, 10, 3)
        first = patch_shuffle(image, patch_size=4, rng=np.random.default_rng(7))
        second = patch_shuffle(image, patch_size=4, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(
            np.sort(first.reshape(-1)), np.sort(image.reshape(-1))
        )
        np.testing.assert_array_equal(first[8:, :, :], image[8:, :, :])
        np.testing.assert_array_equal(first[:, 8:, :], image[:, 8:, :])

    def test_invalid_patch_size(self):
        with self.assertRaises(ValueError):
            patch_shuffle(np.zeros((4, 4, 3), dtype=np.uint8), patch_size=0)

    def test_legacy_remainder_modes_are_explicit(self):
        image = np.ones((10, 10, 3), dtype=np.uint8)
        zeroed = patch_shuffle(
            image, patch_size=4, rng=np.random.default_rng(1), remainder="zero"
        )
        cropped = patch_shuffle(
            image, patch_size=4, rng=np.random.default_rng(1), remainder="crop"
        )
        self.assertEqual(zeroed.shape, image.shape)
        self.assertEqual(cropped.shape, (8, 8, 3))
        self.assertEqual(int(zeroed[8:, :, :].sum()), 0)


if __name__ == "__main__":
    unittest.main()
