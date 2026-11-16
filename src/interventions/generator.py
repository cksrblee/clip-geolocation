"""Generation of the controlled interventions matching 26-3-Summer."""

from __future__ import annotations

import gc
import hashlib
import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .masking import (
    mask_objects_yolo,
    mask_semantic_classes,
    mask_text_easyocr,
    predict_segmentation,
    random_rectangle_control,
)
from .scrambling import patch_mixed_across_regions, patch_shuffle
from .structural import get_canny_edges, get_macro_blur


LOGGER = logging.getLogger(__name__)
VEHICLE_CLASSES = [2, 3, 5, 7]
SKY_CLASSES = [2]
VEGETATION_CLASSES = [4, 9, 17]
BUNDLE_CONDITIONS = {
    "A": (
        "text_masked",
        "vehicle_masked",
        "sky_masked",
        "vegetation_masked",
        "random_rectangle",
        "sky_random_control",
        "vegetation_random_control",
    ),
    "B": ("edges", "macro_blur"),
    "C": (
        "patch_shuffled_16",
        "patch_shuffled_32",
        "patch_shuffled_64",
        "patch_mixed",
    ),
}


def stable_rng(seed: int, filename: str, condition: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{filename}:{condition}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _all_exist(output_dir: Path, filename: str, conditions: Iterable[str]) -> bool:
    return all(
        (output_dir / condition / filename).is_file() for condition in conditions
    )


def _load_image(path: Path) -> tuple[Image.Image, np.ndarray]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return rgb.copy(), np.asarray(rgb).copy()


def build_bundle_a(
    metadata: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    device: str,
    seed: int,
    yolo_weights: str,
    overwrite: bool,
) -> None:
    pending = [
        row
        for row in metadata.itertuples(index=False)
        if overwrite or not _all_exist(output_dir, str(row.filename), BUNDLE_CONDITIONS["A"])
    ]
    if not pending:
        LOGGER.info("Bundle A interventions already exist; skipping generation.")
        return

    import easyocr
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    from ultralytics import YOLO

    reader = easyocr.Reader(["en"], gpu=device.startswith("cuda"))
    yolo = YOLO(yolo_weights)
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    segmenter = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    ).to(device)
    segmenter.eval()

    for row in tqdm(pending, total=len(pending), desc="Bundle A"):
        filename = str(row.filename)
        image_pil, image_np = _load_image(image_dir / filename)

        text_image, text_mask = mask_text_easyocr(image_np.copy(), reader)
        vehicle_image, vehicle_mask = mask_objects_yolo(
            image_np.copy(), yolo, VEHICLE_CLASSES
        )
        avg_area = int((int(text_mask.sum()) + int(vehicle_mask.sum())) // 2)
        random_rect, _ = random_rectangle_control(
            image_np.copy(),
            avg_area,
            stable_rng(seed, filename, "random_rectangle"),
        )

        seg_map = predict_segmentation(image_pil, processor, segmenter, device)
        sky_image, sky_mask = mask_semantic_classes(
            image_np.copy(), seg_map, SKY_CLASSES
        )
        vegetation_image, vegetation_mask = mask_semantic_classes(
            image_np.copy(), seg_map, VEGETATION_CLASSES
        )

        sky_control, _ = random_rectangle_control(
            image_np.copy(),
            int(sky_mask.sum()),
            stable_rng(seed, filename, "sky_random_control"),
        )
        vegetation_control, _ = random_rectangle_control(
            image_np.copy(),
            int(vegetation_mask.sum()),
            stable_rng(seed, filename, "vegetation_random_control"),
        )

        outputs = {
            "text_masked": text_image,
            "vehicle_masked": vehicle_image,
            "sky_masked": sky_image,
            "vegetation_masked": vegetation_image,
            "random_rectangle": random_rect,
            "sky_random_control": sky_control,
            "vegetation_random_control": vegetation_control,
        }
        for condition, image in outputs.items():
            _save_image(image, output_dir / condition / filename)

    del reader, yolo, processor, segmenter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_bundle_b(
    metadata: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    for row in tqdm(
        metadata.itertuples(index=False), total=len(metadata), desc="Bundle B"
    ):
        filename = str(row.filename)
        if not overwrite and _all_exist(output_dir, filename, BUNDLE_CONDITIONS["B"]):
            continue
        _, image_np = _load_image(image_dir / filename)
        _save_image(get_canny_edges(image_np), output_dir / "edges" / filename)
        _save_image(
            get_macro_blur(image_np, sigma=15), output_dir / "macro_blur" / filename
        )


def build_bundle_c(
    metadata: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    seed: int,
    overwrite: bool,
) -> None:
    df_reset = metadata.reset_index(drop=True)
    df_shuffled = df_reset.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    for idx, row in tqdm(
        df_reset.iterrows(), total=len(df_reset), desc="Bundle C"
    ):
        filename = str(row["filename"])
        if not overwrite and _all_exist(output_dir, filename, BUNDLE_CONDITIONS["C"]):
            continue

        _, image_np = _load_image(image_dir / filename)
        image_np_224 = cv2.resize(image_np, (224, 224), interpolation=cv2.INTER_CUBIC)

        for patch_size in (16, 32, 64):
            condition = f"patch_shuffled_{patch_size}"
            shuffled = patch_shuffle(
                image_np_224,
                patch_size,
                stable_rng(seed, filename, condition),
                remainder="zero",
            )
            _save_image(shuffled, output_dir / condition / filename)

        partner_rng = stable_rng(seed, filename, "patch_mixed_partner")
        other_regions = df_shuffled[df_shuffled["region_id"] != row["region_id"]]
        if not other_regions.empty:
            partner_idx = int(partner_rng.integers(0, len(other_regions)))
            partner_filename = str(other_regions.iloc[partner_idx]["filename"])
        else:
            partner_filename = str(df_shuffled.iloc[(idx + 1) % len(df_shuffled)]["filename"])
        _, partner_np = _load_image(image_dir / partner_filename)
        partner_np_224 = cv2.resize(
            partner_np, (224, 224), interpolation=cv2.INTER_CUBIC
        )
        mixed = patch_mixed_across_regions(image_np_224, partner_np_224, patch_size=32)
        _save_image(mixed, output_dir / "patch_mixed" / filename)


def build_interventions(
    metadata: pd.DataFrame,
    image_dir: str | Path,
    output_dir: str | Path,
    bundles: Iterable[str] = ("A", "B", "C"),
    device: str = "cuda",
    seed: int = 42,
    yolo_weights: str = "yolov8n.pt",
    overwrite: bool = False,
) -> None:
    selected = tuple(dict.fromkeys(bundle.upper() for bundle in bundles))
    invalid = sorted(set(selected) - set(BUNDLE_CONDITIONS))
    if invalid:
        raise ValueError(f"Unknown intervention bundles: {', '.join(invalid)}")
    images = Path(image_dir)
    output = Path(output_dir)
    for bundle in selected:
        for condition in BUNDLE_CONDITIONS[bundle]:
            (output / condition).mkdir(parents=True, exist_ok=True)
    if "A" in selected:
        build_bundle_a(metadata, images, output, device, seed, yolo_weights, overwrite)
    if "B" in selected:
        build_bundle_b(metadata, images, output, overwrite)
    if "C" in selected:
        build_bundle_c(metadata, images, output, seed, overwrite)
