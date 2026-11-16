import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.evaluation.evaluator import classification_logits
from src.models.factory import setup_model_strategy
from src.models.lora import TARGET_MODULES, setup_lora_vit


class _DummyVisual(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.output_dim = dim
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim)])
        self.ln_post = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer.resblocks:
            x = block(x)
        return self.ln_post(x)


class _DummyClip(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.visual = _DummyVisual(dim)
        self.text_proj = nn.Linear(dim, dim)
        self.logit_scale = nn.Parameter(torch.tensor(4.6052))

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


class StrategyAndLoRATests(unittest.TestCase):
    def test_pu_and_full_ft_use_text_feature_similarity_without_extra_head(self):
        text_features = torch.eye(3, 8)
        inputs = torch.randn(2, 8)

        pu_model = setup_model_strategy(_DummyClip(), "PU", text_features=text_features, model_name="ViT-L-14")
        self.assertFalse(hasattr(pu_model, "classifier"))
        self.assertFalse(pu_model.visual.transformer.resblocks[0].weight.requires_grad)
        self.assertTrue(pu_model.visual.transformer.resblocks[-1].weight.requires_grad)
        self.assertTrue(pu_model.visual.ln_post.weight.requires_grad)
        self.assertFalse(pu_model.text_proj.weight.requires_grad)

        logits = classification_logits(pu_model, "PU", inputs, text_features)
        self.assertEqual(tuple(logits.shape), (2, 3))

        ft_model = setup_model_strategy(_DummyClip(), "Full-FT", text_features=text_features, model_name="ViT-L-14")
        self.assertTrue(ft_model.visual.transformer.resblocks[0].weight.requires_grad)
        self.assertFalse(ft_model.text_proj.weight.requires_grad)

    def test_peft_lora_configuration(self):
        captured = {}

        class FakeLoraConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        def fake_get_peft_model(module, config):
            captured["wrapped"] = module
            captured["config"] = config
            for p in module.parameters():
                p.requires_grad = True
            return module

        fake_peft = types.SimpleNamespace(
            LoraConfig=FakeLoraConfig, get_peft_model=fake_get_peft_model
        )
        clip = _DummyClip()
        with patch.dict(sys.modules, {"peft": fake_peft}):
            adapted = setup_lora_vit(clip, r=8, alpha=16, dropout=0.1)

        self.assertIs(adapted, clip)
        self.assertEqual(captured["r"], 8)
        self.assertEqual(captured["lora_alpha"], 16)
        self.assertEqual(captured["target_modules"], TARGET_MODULES)
        self.assertEqual(captured["bias"], "none")


if __name__ == "__main__":
    unittest.main()
