import unittest

import torch
from torch import nn

from src.models.lora import LoRALinear, LoRAMultiheadAttention


class LoRAProjectionTests(unittest.TestCase):
    def test_linear_adapter_is_identity_at_initialization_and_receives_gradient(self):
        base = nn.Linear(6, 5)
        adapter = LoRALinear(base, rank=2, alpha=4, dropout=0.0)
        inputs = torch.randn(3, 6)
        torch.testing.assert_close(adapter(inputs), base(inputs))
        adapter(inputs).sum().backward()
        self.assertIsNotNone(adapter.adapter.b.weight.grad)
        self.assertGreater(float(adapter.adapter.b.weight.grad.abs().sum()), 0.0)

    def test_attention_adapters_are_active(self):
        base = nn.MultiheadAttention(8, 2, dropout=0.0)
        adapter = LoRAMultiheadAttention(base, rank=2, alpha=4, dropout=0.0)
        inputs = torch.randn(5, 3, 8)
        expected, _ = base(inputs, inputs, inputs, need_weights=False)
        actual, _ = adapter(inputs, inputs, inputs, need_weights=False)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.sum().backward()
        for branch in (
            adapter.q_adapter,
            adapter.k_adapter,
            adapter.v_adapter,
            adapter.out_adapter,
        ):
            self.assertIsNotNone(branch.b.weight.grad)
            self.assertGreater(float(branch.b.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
