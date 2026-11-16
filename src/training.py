"""Reusable training helpers for the six adaptation strategies."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluation.evaluator import classification_logits, predict_dataset
from .evaluation.metrics import top1_accuracy
from .models.factory import trainable_parameters


LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_text_features(model, tokenizer, prompts: list[str], device: str):
    tokens = tokenizer(prompts).to(device)
    model.eval()
    with torch.inference_mode():
        features = model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)


def save_checkpoint(
    model, checkpoint_path: str | Path, metadata: dict[str, Any] | None = None
) -> None:
    """Save standard model.state_dict() directly for compatibility with 26-3-Summer checkpoints."""
    del metadata  # retained in signature for caller compatibility
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


def load_checkpoint(model, checkpoint_path: str | Path, device: str) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = payload.get("metadata", {})
    else:
        state_dict = payload
        metadata = {}

    if metadata.get("trainable_only"):
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected checkpoint keys: {', '.join(incompatible.unexpected_keys)}"
            )
    else:
        model.load_state_dict(state_dict)
    return metadata


def train_model(
    model,
    strategy: str,
    train_dataset,
    val_dataset,
    text_features,
    checkpoint_path: str | Path,
    device: str,
    model_name: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    num_workers: int,
    weight_decay: float = 0.01,
    accum_steps: int = 1,
) -> float:
    if strategy == "Zero-shot":
        raise ValueError("Zero-shot does not have a training phase")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters(model), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    best_accuracy = -1.0

    eval_batch_size = 16 if "ViT-L" in model_name else 64

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"{strategy} {epoch}/{epochs}"), start=1
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                logits = classification_logits(model, strategy, images, text_features)
                loss = criterion(logits, labels) / accum_steps
            scaler.scale(loss).backward()
            if step % accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * accum_steps
        scheduler.step()

        val_predictions, val_targets = predict_dataset(
            model,
            val_dataset,
            strategy,
            text_features,
            device,
            batch_size=eval_batch_size,
            num_workers=num_workers,
        )
        val_accuracy = top1_accuracy(val_predictions, val_targets)
        LOGGER.info(
            "%s epoch %d/%d: loss=%.4f val_top1=%.2f%%",
            strategy,
            epoch,
            epochs,
            running_loss / max(len(train_loader), 1),
            val_accuracy,
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(model, checkpoint_path)

    load_checkpoint(model, checkpoint_path, device)
    return best_accuracy
