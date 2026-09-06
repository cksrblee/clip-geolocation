#!/usr/bin/env python3
"""Evaluate all adaptation strategies on the 14 controlled probing conditions."""

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
    accuracy_retention,
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
    return parser.parse_args()


def upsert(path: Path, dataframe: pd.DataFrame, keys: list[str]) -> None:
    if path.is_file():
        dataframe = pd.concat([pd.read_csv(path), dataframe], ignore_index=True)
    dataframe = dataframe.drop_duplicates(subset=keys, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


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
    regions = load_region_data(resolve_path(REPO_ROOT, args.regions))
    region_to_id = region_index(regions)
    prompts = region_prompts(regions)

    require_image_files(metadata, image_dir)
    for spec in INTERVENTIONS:
        require_image_files(metadata, interventions_dir / spec.name)

    summary_rows: list[dict] = []
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
        summary_rows.append(
            {
                "Model": args.model,
                "Strategy": strategy,
                "Seed": args.seed,
                "Bundle": "clean",
                "Intervention": "original",
                "Accuracy": clean_accuracy,
                "Accuracy_Drop_pct": 0.0,
                "Switch_Rate_pct": 0.0,
                "Accuracy_Retention": 1.0,
                "Primary_Metric": "Accuracy",
                "Primary_Value": clean_accuracy,
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

        for spec in INTERVENTIONS:
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
            switch_rate = prediction_switch_rate(clean_predictions, predictions)
            retention = accuracy_retention(accuracy, clean_accuracy)
            primary_value = retention if spec.metric == "SS" else switch_rate
            summary_rows.append(
                {
                    "Model": args.model,
                    "Strategy": strategy,
                    "Seed": args.seed,
                    "Bundle": spec.bundle,
                    "Intervention": spec.name,
                    "Accuracy": accuracy,
                    "Accuracy_Drop_pct": clean_accuracy - accuracy,
                    "Switch_Rate_pct": switch_rate,
                    "Accuracy_Retention": retention,
                    "Primary_Metric": spec.metric,
                    "Primary_Value": primary_value,
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

    summary = pd.DataFrame(summary_rows)
    zero_shot = summary[summary["Strategy"] == "Zero-shot"].set_index("Intervention")
    summary["Gain_vs_ZeroShot"] = summary.apply(
        lambda row: (
            row["Primary_Value"] - zero_shot.loc[row["Intervention"], "Primary_Value"]
            if row["Intervention"] in zero_shot.index
            else float("nan")
        ),
        axis=1,
    )
    raw_path = results_dir / "probing_evaluation_results_raw.csv"
    upsert(raw_path, summary, ["Model", "Strategy", "Seed", "Intervention"])
    metrics = summary[summary["Intervention"] != "original"].copy()
    upsert(
        results_dir / "probing_evaluation_results.csv",
        metrics,
        ["Model", "Strategy", "Seed", "Intervention"],
    )
    if prediction_rows:
        upsert(
            results_dir / "probing_predictions.csv",
            pd.DataFrame(prediction_rows),
            ["Model", "Strategy", "Seed", "Intervention", "filename"],
        )


if __name__ == "__main__":
    main()
