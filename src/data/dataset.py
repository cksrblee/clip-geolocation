"""Datasets used by the CLIP adaptation and probing experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class OpenClipDataset(Dataset):
    """Load labeled RGB images without silently replacing failed samples."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str | Path,
        transform: Callable,
        region_to_id: Mapping[str, int],
    ) -> None:
        missing = {"filename", "region_id"} - set(dataframe.columns)
        if missing:
            raise ValueError(
                f"Dataset metadata is missing: {', '.join(sorted(missing))}"
            )
        unknown = sorted(set(dataframe["region_id"]) - set(region_to_id))
        if unknown:
            raise ValueError(f"Unknown region IDs: {', '.join(unknown)}")

        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.region_to_id = dict(region_to_id)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image_path = self.image_dir / str(row["filename"])
        try:
            with Image.open(image_path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc
        return tensor, self.region_to_id[str(row["region_id"])]


def require_image_files(dataframe: pd.DataFrame, image_dir: str | Path) -> None:
    root = Path(image_dir)
    missing = [
        str(name) for name in dataframe["filename"] if not (root / str(name)).is_file()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} images in {root}: {preview}{suffix}"
        )
