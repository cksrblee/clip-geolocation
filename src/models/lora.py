"""LoRA adapters for OpenCLIP ViT attention and MLP projections."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankBranch(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, rank: int, alpha: int, dropout: float
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.a = nn.Linear(input_dim, rank, bias=False)
        self.b = nn.Linear(rank, output_dim, bias=False)
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(self.dropout(inputs))) * self.scaling


class LoRALinear(nn.Module):
    def __init__(self, layer: nn.Linear, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base_layer = layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False
        self.adapter = LowRankBranch(
            layer.in_features, layer.out_features, rank, alpha, dropout
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base_layer(inputs) + self.adapter(inputs)


class LoRAMultiheadAttention(nn.Module):
    """Multi-head attention with active LoRA branches on Q, K, V, and output."""

    def __init__(
        self, layer: nn.MultiheadAttention, rank: int, alpha: int, dropout: float
    ) -> None:
        super().__init__()
        if layer.in_proj_weight is None or layer.kdim != layer.embed_dim:
            raise ValueError("LoRA requires packed equal-dimension Q/K/V projections")
        if layer.vdim != layer.embed_dim or layer.bias_k is not None:
            raise ValueError("Unsupported MultiheadAttention configuration")
        if layer.add_zero_attn:
            raise ValueError("LoRA does not support add_zero_attn")

        self.base_layer = layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False
        dimension = layer.embed_dim
        self.q_adapter = LowRankBranch(dimension, dimension, rank, alpha, dropout)
        self.k_adapter = LowRankBranch(dimension, dimension, rank, alpha, dropout)
        self.v_adapter = LowRankBranch(dimension, dimension, rank, alpha, dropout)
        self.out_adapter = LowRankBranch(dimension, dimension, rank, alpha, dropout)

    def _projection(
        self, inputs: torch.Tensor, offset: int, adapter: LowRankBranch
    ) -> torch.Tensor:
        dimension = self.base_layer.embed_dim
        weight = self.base_layer.in_proj_weight[offset : offset + dimension]
        bias = self.base_layer.in_proj_bias
        projected_bias = None if bias is None else bias[offset : offset + dimension]
        return F.linear(inputs, weight, projected_bias) + adapter(inputs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        unbatched = query.ndim == 2
        if unbatched:
            query, key, value = query[:, None], key[:, None], value[:, None]
        elif self.base_layer.batch_first:
            query, key, value = (
                query.transpose(0, 1),
                key.transpose(0, 1),
                value.transpose(0, 1),
            )

        target_length, batch_size, dimension = query.shape
        source_length = key.shape[0]
        heads = self.base_layer.num_heads
        head_dim = dimension // heads

        q = self._projection(query, 0, self.q_adapter)
        k = self._projection(key, dimension, self.k_adapter)
        v = self._projection(value, 2 * dimension, self.v_adapter)
        q = q.reshape(target_length, batch_size, heads, head_dim).permute(1, 2, 0, 3)
        k = k.reshape(source_length, batch_size, heads, head_dim).permute(1, 2, 0, 3)
        v = v.reshape(source_length, batch_size, heads, head_dim).permute(1, 2, 0, 3)

        scores = torch.matmul(q * (head_dim**-0.5), k.transpose(-2, -1))
        if is_causal and attn_mask is None:
            attn_mask = torch.ones(
                target_length,
                source_length,
                dtype=torch.bool,
                device=query.device,
            ).triu(1)
        if attn_mask is not None:
            mask = attn_mask
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3 and mask.shape[0] == batch_size * heads:
                mask = mask.reshape(batch_size, heads, target_length, source_length)
            elif mask.ndim == 3:
                mask = mask[:, None]
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, float("-inf"))
            else:
                scores = scores + mask.to(dtype=scores.dtype)
        if key_padding_mask is not None:
            padding = key_padding_mask
            if padding.ndim == 1:
                padding = padding[None]
            padding = padding[:, None, None, :]
            if padding.dtype == torch.bool:
                scores = scores.masked_fill(padding, float("-inf"))
            else:
                scores = scores + padding.to(dtype=scores.dtype)

        weights = F.softmax(scores, dim=-1)
        dropped_weights = F.dropout(
            weights, p=self.base_layer.dropout, training=self.training
        )
        context = torch.matmul(dropped_weights, v)
        context = context.permute(2, 0, 1, 3).reshape(
            target_length, batch_size, dimension
        )
        output = self.base_layer.out_proj(context) + self.out_adapter(context)

        if self.base_layer.batch_first and not unbatched:
            output = output.transpose(0, 1)
        if unbatched:
            output = output[:, 0]

        returned_weights: torch.Tensor | None = None
        if need_weights:
            returned_weights = weights.mean(dim=1) if average_attn_weights else weights
            if unbatched:
                returned_weights = returned_weights[0]
        return output, returned_weights


def setup_lora_vit(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
) -> nn.Module:
    """Insert rank-r LoRA branches into every ViT attention and MLP block."""
    if r <= 0:
        raise ValueError("LoRA rank must be positive")
    for parameter in model.parameters():
        parameter.requires_grad = False

    try:
        blocks = model.visual.transformer.resblocks
    except AttributeError as exc:
        raise ValueError("The visual encoder is not an OpenCLIP ViT") from exc
    if not blocks:
        raise ValueError("The visual transformer has no residual blocks")

    for block in blocks:
        if not isinstance(block.attn, nn.MultiheadAttention):
            raise ValueError("Unexpected attention implementation in OpenCLIP ViT")
        if not isinstance(block.mlp.c_fc, nn.Linear) or not isinstance(
            block.mlp.c_proj, nn.Linear
        ):
            raise ValueError("Unexpected MLP implementation in OpenCLIP ViT")
        block.attn = LoRAMultiheadAttention(block.attn, r, alpha, dropout)
        block.mlp.c_fc = LoRALinear(block.mlp.c_fc, r, alpha, dropout)
        block.mlp.c_proj = LoRALinear(block.mlp.c_proj, r, alpha, dropout)

    if not any(parameter.requires_grad for parameter in model.visual.parameters()):
        raise RuntimeError("LoRA setup produced no trainable parameters")
    return model
