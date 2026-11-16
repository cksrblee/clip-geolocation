import unittest

from src.experiment import (
    EVALUATION_CONDITIONS,
    INTERVENTIONS,
    STRATEGIES,
    region_index,
)


class ExperimentDefinitionTests(unittest.TestCase):
    def test_paper_strategy_set_excludes_contrastive(self):
        self.assertEqual(
            STRATEGIES,
            ("Zero-shot", "LP-T", "LP-C", "PU", "LoRA", "Full-FT"),
        )

    def test_clean_plus_thirteen_interventions(self):
        self.assertEqual(len(EVALUATION_CONDITIONS), 14)
        self.assertEqual(len({spec.name for spec in INTERVENTIONS}), 13)
        self.assertEqual(EVALUATION_CONDITIONS[0], "original")

    def test_interventions_match_original_probing_suite(self):
        controls = {
            spec.name: spec.control
            for spec in INTERVENTIONS
            if spec.control is not None
        }
        self.assertEqual(
            controls,
            {
                "text_masked": "random_rectangle",
                "vehicle_masked": "random_rectangle",
                "sky_masked": "sky_random_control",
                "vegetation_masked": "vegetation_random_control",
            },
        )
        self.assertIn("patch_mixed", EVALUATION_CONDITIONS)

    def test_region_index_preserves_config_order(self):
        regions = [{"id": "z"}, {"id": "a"}]
        self.assertEqual(region_index(regions), {"z": 0, "a": 1})


if __name__ == "__main__":
    unittest.main()
