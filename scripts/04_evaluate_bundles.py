#!/usr/bin/env python3
"""Evaluate adaptation strategies on the controlled probing interventions (Tables I-II, Fig. 3)."""

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
    calculate_shortcut_reliance,
    calculate_structure_sufficiency,
    load_region_data,
    prediction_switch_rate,
    top1_accuracy,
)
from src.experiment import (
    INTERVENTIONS,
    STRATEGIES,
    checkpoint_filename,
    region_index,
    region_prompts,
    resolve_path,
)
from src.models.factory import setup_model_strategy
from src.models.openclip import create_model_and_transforms
from src.training import encode_text_features, load_checkpoint, set_seed


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES)
    )
    parser.add_argument("--metadata", default="dataset/val_metadata.csv")
    parser.add_argument("--image-dir", default="dataset/images")
    parser.add_argument("--interventions-dir", default="interventions/val")
    parser.add_argument("--regions", default="configs/la_county_regions.json")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip intervention folders that have not been generated yet",
    )
    return parser.parse_args()


def merge_table(path: Path, dataframe: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if path.is_file():
        dataframe = pd.concat([pd.read_csv(path), dataframe], ignore_index=True)
    return dataframe.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def compute_gains(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    zs_rows = result[result["Strategy"] == "Zero-shot"]
    zs_lookup = {
        (row["Model"], row["Seed"], row["Intervention"]): row["Value"]
        for _, row in zs_rows.iterrows()
    }
    result["Gain"] = result.apply(
        lambda row: (
            row["Value"]
            - zs_lookup[(row["Model"], row["Seed"], row["Intervention"])]
            if (row["Model"], row["Seed"], row["Intervention"]) in zs_lookup
            else float("nan")
        ),
        axis=1,
    )
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metadata = pd.read_csv(resolve_path(REPO_ROOT, args.metadata))
    image_dir = resolve_path(REPO_ROOT, args.image_dir)
    interventions_dir = resolve_path(REPO_ROOT, args.interventions_dir)
    checkpoints_dir = resolve_path(REPO_ROOT, args.checkpoints_dir)
    results_dir = resolve_path(REPO_ROOT, args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    regions = load_region_data(resolve_path(REPO_ROOT, args.regions))
    region_to_id = region_index(regions)
    prompts = region_prompts(regions)

    require_image_files(metadata, image_dir)
    active_interventions = []
    for spec in INTERVENTIONS:
        cond_dir = interventions_dir / spec.name
        if not cond_dir.is_dir() and args.allow_missing:
            LOGGER.info("Skipping %s (directory not found)", spec.name)
            continue
        require_image_files(metadata, cond_dir)
        active_interventions.append(spec)

    raw_rows: list[dict] = []
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for strategy in args.strategies:
        LOGGER.info("Evaluating %s / %s", args.model, strategy)
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
        if strategy != "Zero-shot":
            checkpoint_path = checkpoints_dir / checkpoint_filename(
                args.model, strategy, args.seed
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Missing checkpoint for {strategy}: {checkpoint_path}. Run 01_train_baselines.py first."
                )
            load_checkpoint(model, checkpoint_path, device)

        clean_dataset = OpenClipDataset(metadata, image_dir, preprocess, region_to_id)
        clean_predictions, targets = predict_dataset(
            model,
            clean_dataset,
            strategy,
            text_features,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        clean_accuracy = top1_accuracy(clean_predictions, targets)
        raw_rows.append(
            {
                "Model": args.model,
                "Strategy": strategy,
                "Seed": args.seed,
                "Intervention": "original",
                "Accuracy": clean_accuracy,
            }
        )

        if args.save_predictions:
            for filename, target, prediction in zip(
                metadata["filename"], targets, clean_predictions
            ):
                prediction_rows.append(
                    {
                        "Model": args.model,
                        "Strategy": strategy,
                        "Seed": args.seed,
                        "Intervention": "original",
                        "filename": filename,
                        "Target": int(target),
                        "Prediction": int(prediction),
                    }
                )

        for spec in active_interventions:
            dataset = OpenClipDataset(
                metadata, interventions_dir / spec.name, preprocess, region_to_id
            )
            predictions, intervention_targets = predict_dataset(
                model,
                dataset,
                strategy,
                text_features,
                device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            if not (targets == intervention_targets).all():
                raise RuntimeError(f"Target ordering changed for {spec.name}")
            accuracy = top1_accuracy(predictions, targets)
            sr = calculate_shortcut_reliance(clean_accuracy, accuracy)
            ss = calculate_structure_sufficiency(accuracy, clean_accuracy)
            switch_rate = prediction_switch_rate(clean_predictions, predictions)
            value = ss if spec.metric == "SS" else sr

            raw_rows.append(
                {
                    "Model": args.model,
                    "Strategy": strategy,
                    "Seed": args.seed,
                    "Intervention": spec.name,
                    "Accuracy": accuracy,
                }
            )
            metric_rows.append(
                {
                    "Model": args.model,
                    "Strategy": strategy,
                    "Seed": args.seed,
                    "Bundle": spec.bundle,
                    "Intervention": spec.name,
                    "Accuracy": accuracy,
                    "Metric": spec.metric,
                    "Value": value,
                    "Prediction_Switch_Rate_pct": switch_rate,
                }
            )
            if args.save_predictions:
                for filename, target, prediction in zip(
                    metadata["filename"], targets, predictions
                ):
                    prediction_rows.append(
                        {
                            "Model": args.model,
                            "Strategy": strategy,
                            "Seed": args.seed,
                            "Intervention": spec.name,
                            "filename": filename,
                            "Target": int(target),
                            "Prediction": int(prediction),
                        }
                    )

        del model, clip_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_path = results_dir / "probing_evaluation_results_raw.csv"
    merged_raw = merge_table(
        raw_path,
        pd.DataFrame(raw_rows),
        ["Model", "Strategy", "Seed", "Intervention"],
    )
    merged_raw.to_csv(raw_path, index=False)

    metrics_path = results_dir / "probing_evaluation_results.csv"
    merged_metrics = merge_table(
        metrics_path,
        pd.DataFrame(metric_rows),
        ["Model", "Strategy", "Seed", "Intervention"],
    )
    merged_metrics = compute_gains(merged_metrics)
    merged_metrics.to_csv(metrics_path, index=False)

    if prediction_rows:
        pred_path = results_dir / "probing_predictions.csv"
        merged_preds = merge_table(
            pred_path,
            pd.DataFrame(prediction_rows),
            ["Model", "Strategy", "Seed", "Intervention", "filename"],
        )
        merged_preds.to_csv(pred_path, index=False)


if __name__ == "__main__":
    main()
