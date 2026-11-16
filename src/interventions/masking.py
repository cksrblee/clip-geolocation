"""
Bundle A Interventions: Visual Shortcut Removal and Inpainting.
Includes OCR text masking, YOLO vehicle masking, SegFormer sky/vegetation masking,
and random rectangle control masking matching 26-3-Summer/02_build_interventions.py.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch


def apply_mask_and_inpaint(
    img_np: np.ndarray, mask: np.ndarray, radius: int = 3
) -> np.ndarray:
    """Inpaint masked regions with the Telea algorithm."""
    if mask.shape != img_np.shape[:2]:
        raise ValueError("Mask shape must match image height and width")
    mask_uint8 = mask.astype(bool).astype(np.uint8) * 255
    inpainted = cv2.inpaint(
        img_np, mask_uint8, inpaintRadius=radius, flags=cv2.INPAINT_TELEA
    )
    return inpainted


def mask_text_easyocr(img_np: np.ndarray, reader) -> Tuple[np.ndarray, np.ndarray]:
    """Detect text bounding boxes using EasyOCR and inpaint them."""
    results = reader.readtext(img_np)
    mask = np.zeros(img_np.shape[:2], dtype=bool)
    for bbox, _, _ in results:
        points = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask.view(np.uint8), [points], 1)
    inpainted = apply_mask_and_inpaint(img_np, mask)
    return inpainted, mask


def mask_objects_yolo(
    img_np: np.ndarray, yolo_model, target_classes: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect requested COCO object classes with YOLOv8 and inpaint them."""
    results = yolo_model(img_np, verbose=False)[0]
    mask = np.zeros(img_np.shape[:2], dtype=bool)
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id in target_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, x2 = np.clip([x1, x2], 0, img_np.shape[1])
            y1, y2 = np.clip([y1, y2], 0, img_np.shape[0])
            mask[y1:y2, x1:x2] = True
    inpainted = apply_mask_and_inpaint(img_np, mask)
    return inpainted, mask


def predict_segmentation(
    img_pil: Image.Image, processor, model, device: str
) -> np.ndarray:
    inputs = processor(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=img_pil.size[::-1], mode="bilinear", align_corners=False
    )
    return upsampled_logits.argmax(dim=1)[0].cpu().numpy()


def mask_semantic_segformer(
    img_pil: Image.Image,
    processor,
    model,
    target_classes: List[int],
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment semantic classes (sky=2, vegetation=4,9,17) using SegFormer-ADE20K."""
    seg_map = predict_segmentation(img_pil, processor, model, device)
    return mask_semantic_classes(np.array(img_pil), seg_map, target_classes)


def mask_semantic_classes(
    img_np: np.ndarray,
    seg_map: np.ndarray,
    target_classes: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(img_np.shape[:2], dtype=bool)
    for c in target_classes:
        mask = mask | (seg_map == c)
    inpainted = apply_mask_and_inpaint(img_np, mask)
    return inpainted, mask


def random_rectangle_control(
    img_np: np.ndarray,
    target_area: int,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inpaint a random square control region matching 26-3-Summer/02_build_interventions.py."""
    h, w = img_np.shape[:2]
    if target_area <= 0:
        target_area = int(h * w * 0.05)  # Default 5% fallback as in 26-3-Summer

    side = int(np.sqrt(target_area))
    if side >= h or side >= w:
        side = max(1, min(h, w) - 1)
    side = max(1, side)

    generator = rng if rng is not None else np.random.default_rng()
    x1 = int(generator.integers(0, w - side + 1))
    y1 = int(generator.integers(0, h - side + 1))

    mask = np.zeros((h, w), dtype=bool)
    mask[y1 : y1 + side, x1 : x1 + side] = True
    return apply_mask_and_inpaint(img_np, mask), mask
