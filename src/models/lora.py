"""PEFT LoRA setup for OpenCLIP visual encoders."""

from __future__ import annotations

import logging

import torch.nn as nn


LOGGER = logging.getLogger(__name__)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"]


def setup_lora_vit(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
) -> nn.Module:
    """Wrap model.visual with PEFT LoRA adapters on ViT attention and MLP projections."""
    if r <= 0:
        raise ValueError("LoRA rank must be positive")

    from peft import LoraConfig, get_peft_model

    for parameter in model.parameters():
        parameter.requires_grad = False

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=list(TARGET_MODULES),
        lora_dropout=dropout,
        bias="none",
    )
    model.visual = get_peft_model(model.visual, config)
    return model
