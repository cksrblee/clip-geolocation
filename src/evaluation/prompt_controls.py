"""Prompt variants used by the zero-shot grounding control experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..experiment import region_prompts


LENGTH_FILLER = (
    " which is a nice area located in the sprawling city with various buildings, "
    "roads, vehicles, and weather conditions present today."
)


def load_visual_descriptors(
    config_path: str | Path,
    regions: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    descriptors = payload.get("visual_descriptors")
    if not isinstance(descriptors, dict):
        raise ValueError("Prompt config must contain a visual_descriptors object")

    region_ids = [str(region["id"]) for region in regions]
    missing = sorted(set(region_ids) - set(descriptors))
    extra = sorted(set(descriptors) - set(region_ids))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected: {', '.join(extra)}")
        raise ValueError("Descriptor region mismatch (" + "; ".join(details) + ")")

    result: dict[str, str] = {}
    forbidden_names = [str(region["name"]).casefold() for region in regions]
    forbidden_names.extend(("los angeles", "greater los angeles"))
    for region_id in region_ids:
        descriptor = str(descriptors[region_id]).strip()
        if not descriptor:
            raise ValueError(f"Visual descriptor is empty for {region_id}")
        lowered = descriptor.casefold()
        leaked = [name for name in forbidden_names if name in lowered]
        if leaked:
            raise ValueError(
                f"Visual descriptor for {region_id} contains a region name: {leaked[0]}"
            )
        result[region_id] = descriptor
    return result


def build_prompt_controls(
    regions: Sequence[Mapping[str, Any]],
    visual_descriptors: Mapping[str, str],
) -> dict[str, list[str]]:
    original = region_prompts(regions)
    descriptors = [str(visual_descriptors[str(region["id"])]) for region in regions]
    return {
        "P0_original": original,
        "P_len": [prompt + LENGTH_FILLER for prompt in original],
        "P_visual_only": descriptors,
        "P_swap": descriptors[1:] + descriptors[:1],
    }
