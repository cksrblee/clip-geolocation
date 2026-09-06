#!/usr/bin/env python3
"""Evaluate patch scrambling on a 2,000-image Caltech-101 control sample."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
import torchvision
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import top1_accuracy
from src.experiment import resolve_path
from src.interventions.scrambling import patch_shuffle
from src.models.openclip import create_model_and_transforms


def image_rng(seed: int, sample_index: int, patch_size: int) -> np.random.Generator:
    digest = hashlib.sha256(
        f"{seed}:{sample_index}:{patch_size}".encode("ascii")
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--dataset-dir", default="dataset/caltech101")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = torchvision.datasets.Caltech101(
        root=resolve_path(REPO_ROOT, args.dataset_dir), download=True
    )
    sample_count = min(args.samples, len(dataset))
    indices = random.Random(args.seed).sample(range(len(dataset)), sample_count)
    class_prompts = [
        f"a photo of a {name.replace('_', ' ')}" for name in dataset.categories
    ]
    class_prompts.append("a photo of a background scene")

    model, _, preprocess = create_model_and_transforms(
        args.model, args.pretrained, device
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()
    with torch.inference_mode():
        text_features = model.encode_text(tokenizer(class_prompts).to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    rows: list[dict] = []
    clean_accuracy: float | None = None
    for patch_size in (0, 16, 32, 64):
        predictions: list[int] = []
        targets: list[int] = []
        for start in tqdm(
            range(0, len(indices), args.batch_size),
            desc=f"Patch-{patch_size or 'clean'}",
        ):
            batch_indices = indices[start : start + args.batch_size]
            tensors = []
            for index in batch_indices:
                image, target = dataset[index]
                image = image.convert("RGB").resize(
                    (224, 224), Image.Resampling.BICUBIC
                )
                if patch_size:
                    shuffled = patch_shuffle(
                        np.asarray(image),
                        patch_size,
                        image_rng(args.seed, index, patch_size),
                        remainder="preserve",
                    )
                    image = Image.fromarray(shuffled)
                tensors.append(preprocess(image))
                targets.append(int(target))
            batch = torch.stack(tensors).to(device)
            with torch.inference_mode():
                features = model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
                predictions.extend(
                    (features @ text_features.T).argmax(dim=1).cpu().tolist()
                )

        accuracy = top1_accuracy(predictions, targets)
        if patch_size == 0:
            clean_accuracy = accuracy
        assert clean_accuracy is not None
        rows.append(
            {
                "Dataset": "caltech101",
                "Model": args.model,
                "Condition": "Clean" if patch_size == 0 else f"Patch-{patch_size}",
                "Accuracy": accuracy,
                "Relative_Drop_pct": 100.0
                * (clean_accuracy - accuracy)
                / clean_accuracy,
                "Samples": sample_count,
                "Classes": len(class_prompts),
                "Seed": args.seed,
            }
        )

    results_dir = resolve_path(REPO_ROOT, args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sampled_index": indices}).to_csv(
        results_dir / f"caltech101_sampled_indices_seed{args.seed}.csv", index=False
    )
    pd.DataFrame(rows).to_csv(
        results_dir / "caltech101_control_results.csv", index=False
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
