#!/usr/bin/env python3
"""Evaluate the four zero-shot prompt controls reported in Table IX."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import open_clip
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import OpenClipDataset, require_image_files
from src.evaluation.metrics import load_region_data, top1_accuracy
from src.evaluation.prompt_controls import (
    build_prompt_controls,
    load_visual_descriptors,
)
from src.experiment import region_index, resolve_path
from src.models.openclip import create_model_and_transforms


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--metadata", default="dataset/test_metadata.csv")
    parser.add_argument("--image-dir", default="dataset/images")
    parser.add_argument("--regions", default="configs/la_county_regions.json")
    parser.add_argument("--prompt-config", default="configs/prompt_controls.json")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metadata = pd.read_csv(resolve_path(REPO_ROOT, args.metadata))
    image_dir = resolve_path(REPO_ROOT, args.image_dir)
    require_image_files(metadata, image_dir)
    regions = load_region_data(resolve_path(REPO_ROOT, args.regions))
    region_to_id = region_index(regions)
    visual_descriptors = load_visual_descriptors(
        resolve_path(REPO_ROOT, args.prompt_config), regions
    )
    prompt_controls = build_prompt_controls(regions, visual_descriptors)

    model, _, preprocess = create_model_and_transforms(
        args.model, args.pretrained, device
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()
    dataset = OpenClipDataset(metadata, image_dir, preprocess, region_to_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    image_batches = []
    label_batches = []
    with torch.inference_mode():
        for images, labels in loader:
            features = model.encode_image(images.to(device, non_blocking=True))
            image_batches.append(features / features.norm(dim=-1, keepdim=True))
            label_batches.append(labels)
    image_features = torch.cat(image_batches)
    labels = torch.cat(label_batches).numpy()

    results: list[dict] = []
    for variant, prompts in prompt_controls.items():
        tokens = tokenizer(prompts).to(device)
        with torch.inference_mode():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            predictions = (image_features @ text_features.T).argmax(dim=1).cpu().numpy()
        accuracy = top1_accuracy(predictions, labels)
        LOGGER.info("%s: %.2f%%", variant, accuracy)
        results.append(
            {
                "Model": args.model,
                "Prompt_Variant": variant,
                "Accuracy": accuracy,
                "Chance_Accuracy": 100.0 / len(regions),
                "Samples": len(metadata),
            }
        )

    results_dir = resolve_path(REPO_ROOT, args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(
        results_dir / "prompt_grounding_results.csv", index=False
    )
    prompt_rows = []
    for variant, prompts in prompt_controls.items():
        for region, prompt in zip(regions, prompts):
            prompt_rows.append(
                {
                    "Prompt_Variant": variant,
                    "Region_ID": region["id"],
                    "Prompt": prompt,
                }
            )
    pd.DataFrame(prompt_rows).to_csv(
        results_dir / "prompt_grounding_prompts.csv", index=False
    )


if __name__ == "__main__":
    main()
