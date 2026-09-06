import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.prompt_controls import (
    build_prompt_controls,
    load_visual_descriptors,
)


class PromptControlTests(unittest.TestCase):
    def setUp(self):
        self.regions = [
            {"id": "a", "name": "Alpha"},
            {"id": "b", "name": "Beta"},
        ]
        self.descriptors = {
            "a": "tall buildings and broad streets",
            "b": "low buildings and mature trees",
        }

    def test_four_conditions_and_cyclic_swap(self):
        controls = build_prompt_controls(self.regions, self.descriptors)
        self.assertEqual(
            tuple(controls),
            ("P0_original", "P_len", "P_visual_only", "P_swap"),
        )
        self.assertEqual(controls["P_visual_only"], list(self.descriptors.values()))
        self.assertEqual(
            controls["P_swap"],
            controls["P_visual_only"][1:] + controls["P_visual_only"][:1],
        )

    def test_descriptor_loader_rejects_region_name_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(
                json.dumps(
                    {
                        "visual_descriptors": {
                            "a": "a street in Alpha",
                            "b": "low buildings",
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "contains a region name"):
                load_visual_descriptors(path, self.regions)


if __name__ == "__main__":
    unittest.main()
