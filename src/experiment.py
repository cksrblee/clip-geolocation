"""Canonical definitions shared by the paper experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STRATEGIES = ("Zero-shot", "LP-T", "LP-C", "PU", "LoRA", "Full-FT")
PROMPT_TEMPLATE = "a Google Street View photo in {name}, Los Angeles"


@dataclass(frozen=True)
class InterventionSpec:
    name: str
    bundle: str
    metric: str
    control: str | None = None


# The clean image plus these 13 interventions are the 14 evaluated conditions.
INTERVENTIONS = (
    InterventionSpec("text_masked", "A", "SR", "text_random_control"),
    InterventionSpec("vehicle_masked", "A", "SR", "vehicle_random_control"),
    InterventionSpec("sky_masked", "A", "SR", "sky_random_control"),
    InterventionSpec("vegetation_masked", "A", "SR", "vegetation_random_control"),
    InterventionSpec("text_random_control", "A-control", "SR"),
    InterventionSpec("vehicle_random_control", "A-control", "SR"),
    InterventionSpec("sky_random_control", "A-control", "SR"),
    InterventionSpec("vegetation_random_control", "A-control", "SR"),
    InterventionSpec("edges", "B", "SS"),
    InterventionSpec("macro_blur", "B", "SS"),
    InterventionSpec("patch_shuffled_16", "C", "SR"),
    InterventionSpec("patch_shuffled_32", "C", "SR"),
    InterventionSpec("patch_shuffled_64", "C", "SR"),
)

INTERVENTION_BY_NAME = {spec.name: spec for spec in INTERVENTIONS}
EVALUATION_CONDITIONS = ("original", *(spec.name for spec in INTERVENTIONS))


def ordered_regions(
    region_data: Iterable[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Keep JSON order as the single class-index order used by every experiment."""
    regions = list(region_data)
    ids = [str(region["id"]) for region in regions]
    if len(ids) != len(set(ids)):
        raise ValueError("Region IDs must be unique")
    return regions


def region_index(region_data: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return {
        str(region["id"]): index
        for index, region in enumerate(ordered_regions(region_data))
    }


def region_prompts(region_data: Iterable[Mapping[str, Any]]) -> list[str]:
    return [
        PROMPT_TEMPLATE.format(name=region["name"])
        for region in ordered_regions(region_data)
    ]


def checkpoint_filename(model_name: str, strategy: str, seed: int) -> str:
    safe_model_name = model_name.replace("/", "_")
    return f"{safe_model_name}_{strategy}_seed{seed}.pt"


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def validate_strategies(strategies: Sequence[str]) -> tuple[str, ...]:
    unknown = sorted(set(strategies) - set(STRATEGIES))
    if unknown:
        raise ValueError(f"Unknown adaptation strategies: {', '.join(unknown)}")
    return tuple(strategies)
