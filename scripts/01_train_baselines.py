#!/usr/bin/env python3
"""Train and evaluate the six adaptation strategies reported in Tables III-IV."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import open_clip
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import OpenClipDataset, require_image_files
from src.evaluation.evaluator import predict_dataset
from src.evaluation.metrics import (
    calculate_metrics,
    get_region_centers,
    load_region_data,
    top1_accuracy,
)
from src.experiment import (
    STRATEGIES,
    checkpoint_filename,
    region_index,
    region_prompts,
    resolve_path,
)
from src.models.factory import setup_model_strategy
from src.models.openclip import create_model_and_transforms
from src.training import encode_text_features, load_checkpoint, set_seed, train_model


LOGGER = logging.getLogger(__name__)
DEFAULT_LEARNING_RATES = {
    "LP-T": 1e-4,
    "LP-C": 1e-4,
    "PU": 1e-5,
    "LoRA": 1e-4,
    "Full-FT": 1e-5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES)
    )
    parser.add_argument("--metadata-dir", default="dataset")
    parser.add_argument("--image-dir", default="dataset/images")
    parser.add_argument("--regions", default="configs/la_county_regions.json")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true", help="Retrain existing checkpoints"
    )
    return parser.parse_args()


def load_split(metadata_dir: Path, name: str) -> pd.DataFrame:
    path = metadata_dir / f"{name}_metadata.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset split: {path}")
    return pd.read_csv(path)


def upsert_results(path: Path, rows: list[dict]) -> None:
    new_results = pd.DataFrame(rows)
    if path.is_file():
        new_results = pd.concat([pd.read_csv(path), new_results], ignore_index=True)
    new_results = new_results.drop_duplicates(
        subset=["Model", "Strategy", "Seed"], keep="last"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    new_results.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metadata_dir = resolve_path(REPO_ROOT, args.metadata_dir)
    image_dir = resolve_path(REPO_ROOT, args.image_dir)
    regions_path = resolve_path(REPO_ROOT, args.regions)
    checkpoints_dir = resolve_path(REPO_ROOT, args.checkpoints_dir)
    results_dir = resolve_path(REPO_ROOT, args.results_dir)

    regions = load_region_data(regions_path)
    region_to_id = region_index(regions)
    id_to_region = {index: region_id for region_id, index in region_to_id.items()}
    centers = get_region_centers(regions)
    prompts = region_prompts(regions)
    splits = {name: load_split(metadata_dir, name) for name in ("train", "val", "test")}
    for split in splits.values():
        require_image_files(split, image_dir)

    result_rows: list[dict] = []
    per_region_rows: list[dict] = []
    for strategy in args.strategies:
        LOGGER.info("Preparing %s / %s", args.model, strategy)
        clip_model, _, preprocess = create_model_and_transforms(
            args.model, args.pretrained, device
        )
        tokenizer = open_clip.get_tokenizer(args.model)
        text_features = encode_text_features(clip_model, tokenizer, prompts, device)
        model = setup_model_strategy(
            clip_model,
            strategy,
            text_features=text_features,
            model_name=args.model,
            num_classes=len(regions),
        ).to(device)

        datasets = {
            name: OpenClipDataset(split, image_dir, preprocess, region_to_id)
            for name, split in splits.items()
        }
        checkpoint_path = checkpoints_dir / checkpoint_filename(
            args.model, strategy, args.seed
        )
        if strategy != "Zero-shot":
            if checkpoint_path.is_file() and not args.force:
                LOGGER.info("Loading existing checkpoint %s", checkpoint_path)
                load_checkpoint(model, checkpoint_path, device)
            else:
                learning_rate = args.learning_rate or DEFAULT_LEARNING_RATES[strategy]
                train_model(
                    model=model,
                    strategy=strategy,
                    train_dataset=datasets["train"],
                    val_dataset=datasets["val"],
                    text_features=text_features,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    model_name=args.model,
                    seed=args.seed,
                    epochs=args.epochs,
                    learning_rate=learning_rate,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )

        predictions, targets = predict_dataset(
            model,
            datasets["test"],
            strategy,
            text_features,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        coordinates = list(zip(splits["test"]["latitude"], splits["test"]["longitude"]))
        metrics = calculate_metrics(
            predictions,
            targets,
            coordinates,
            id_to_region,
            centers,
        )
        LOGGER.info(
            "%s: test_top1=%.2f%% centroid_error=%.2f km",
            strategy,
            metrics["accuracy"],
            metrics["mean_error_km"],
        )
        result_rows.append(
            {
                "Model": args.model,
                "Strategy": strategy,
                "Seed": args.seed,
                "Top1_Accuracy": metrics["accuracy"],
                "Centroid_Error_km": metrics["mean_error_km"],
            }
        )
        for class_index, region in enumerate(regions):
            selected = targets == class_index
            if not selected.any():
                raise RuntimeError(f"Test split has no samples for {region['id']}")
            per_region_rows.append(
                {
                    "Model": args.model,
                    "Strategy": strategy,
                    "Seed": args.seed,
                    "Region_ID": region["id"],
                    "Region": region["name"],
                    "Accuracy": top1_accuracy(predictions[selected], targets[selected]),
                    "Samples": int(selected.sum()),
                }
            )
        del model, clip_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    upsert_results(results_dir / "baseline_results.csv", result_rows)
    per_region = pd.DataFrame(per_region_rows)
    per_region_path = results_dir / "baseline_per_region.csv"
    if per_region_path.is_file():
        per_region = pd.concat(
            [pd.read_csv(per_region_path), per_region], ignore_index=True
        )
    per_region = per_region.drop_duplicates(
        subset=["Model", "Strategy", "Seed", "Region_ID"], keep="last"
    )
    per_region.to_csv(per_region_path, index=False)


if __name__ == "__main__":
    main()
