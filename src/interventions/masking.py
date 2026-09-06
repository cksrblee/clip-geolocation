"""
Bundle A Interventions: Visual Shortcut Removal and Inpainting.
Includes OCR text masking, YOLO vehicle masking, SegFormer sky/vegetation masking,
and random geometric control masking.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from typing import List, Tuple


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


def mask_semantic_segformer(
    img_pil: Image.Image,
    processor,
    model,
    target_classes: List[int],
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment semantic classes (e.g. sky=2, vegetation=4,9,17) using SegFormer-ADE20K."""
    seg_map = predict_segmentation(img_pil, processor, model, device)
    return mask_semantic_classes(np.array(img_pil), seg_map, target_classes)


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
    """Inpaint a deterministic compact random mask with exactly target_area pixels."""
    h, w = img_np.shape[:2]
    if target_area <= 0:
        return img_np.copy(), np.zeros((h, w), dtype=bool)
    target_area = int(np.clip(target_area, 1, h * w))
    minimum_width = int(np.ceil(target_area / h))
    rect_w = min(
        w,
        max(minimum_width, int(np.ceil(np.sqrt(target_area)))),
    )
    full_rows, partial_width = divmod(target_area, rect_w)
    rect_h = full_rows + int(partial_width > 0)
    generator = rng if rng is not None else np.random.default_rng()
    x1 = int(generator.integers(0, w - rect_w + 1))
    y1 = int(generator.integers(0, h - rect_h + 1))

    mask = np.zeros((h, w), dtype=bool)
    if full_rows:
        mask[y1 : y1 + full_rows, x1 : x1 + rect_w] = True
    if partial_width:
        mask[y1 + full_rows, x1 : x1 + partial_width] = True

    return apply_mask_and_inpaint(img_np, mask), mask
