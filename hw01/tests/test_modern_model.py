"""Functional checks for modern GPT, with classic checkpoint compatibility."""
from copy import deepcopy
import math
from pathlib import Path
import unittest
import uuid

import torch
from torch.nn import functional as F

from test_training import fixture, assert_nested_equal, clean_fixture
from poetrygpt.model import GPT, ModelConfig, RotaryEmbedding, token_losses
from poetrygpt.runtime import load_checkpoint, seed_everything
from poetrygpt.train import train


class ModernModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, dropout=0.0):
        return GPT(ModelConfig(vocab_size=32, context_length=32, n_layers=2, n_heads=2,
                               d_model=16, dropout=dropout, architecture='modern', ffn_hidden_size=32))

    def test_rope_matches_pairwise_rotation_and_preserves_relative_dot_products(self):
        seed_everything(2)
        rope = RotaryEmbedding(8, 16)
        x = torch.randn(1, 2, 7, 8, requires_grad=True)
        actual = rope(x)
        expected = x.detach().clone()
        for position in range(7):
            for pair in range(4):
                angle = position / (10000.0 ** (2 * pair / 8))
                a, b = x.detach()[..., position, 2 * pair], x.detach()[..., position, 2 * pair + 1]
                expected[..., position, 2 * pair] = a * math.cos(angle) - b * math.sin(angle)
                expected[..., position, 2 * pair + 1] = a * math.sin(angle) + b * math.cos(angle)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.square().sum(-1), x.square().sum(-1))
        torch.testing.assert_close(actual[..., 0, :], x[..., 0, :])
        q, k = [rope(torch.randn(1, 1, 1, 8).expand(1, 1, 7, 8)) for _ in range(2)]
        torch.testing.assert_close((q[..., 0, :] * k[..., 3, :]).sum(),
                                   (q[..., 2, :] * k[..., 5, :]).sum())
        actual.square().sum().backward()
        torch.testing.assert_close(x.grad, 2 * x.detach())

    def test_modern_causality_right_padding_and_swiglu_gradients(self):
        seed_everything(3)
        model = self.model().eval()
        ids = torch.tensor([[1, 6, 7, 8, 2]])
        changed = ids.clone()
        changed[:, 3:] = torch.tensor([9, 10])
        torch.testing.assert_close(model(ids)[:, :3], model(changed)[:, :3], atol=1e-7, rtol=0)
        padded = F.pad(ids, (0, 3))
        torch.testing.assert_close(model(ids), model(padded)[:, :5], atol=1e-7, rtol=0)
        self.assertIsNone(model.positions)
        self.assertFalse(any('rotary' in name for name in model.state_dict()))
        loss, _ = token_losses(model(ids), ids)
        loss.sum().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
        block = model.blocks[0]
        x = torch.randn(2, 4, 16)
        after_attention = x + block.attention(block.norm1(x))
        norm = block.norm2(after_attention)
        expected = after_attention + block.contract(F.silu(block.gate(norm)) * block.expand(norm))
        torch.testing.assert_close(block(x), expected)
        norm_reference = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * block.norm1.weight
        torch.testing.assert_close(block.norm1(x), norm_reference)

    def test_gradient_checkpointing_preserves_modern_dropout_and_gradients(self):
        seed_everything(4)
        plain = self.model(dropout=0.2)
        checked = deepcopy(plain)
        checked.config.gradient_checkpointing = True
        ids = torch.tensor([[1, 6, 7, 8, 2]])
        seed_everything(5)
        first = token_losses(plain(ids), ids)[0].sum()
        first.backward()
        seed_everything(5)
        second = token_losses(checked(ids), ids)[0].sum()
        second.backward()
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        for left, right in zip(plain.parameters(), checked.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)

    def test_modern_training_exact_resume_and_classic_defaults(self):
        folder = Path('audit/training_tests_tmp') / uuid.uuid4().hex
        cfg = fixture(folder / 'data')
        self.addCleanup(clean_fixture, folder)
        cfg['model'].update(architecture='modern', ffn_hidden_size=32)
        full = load_checkpoint(train(cfg, folder / 'full', 'cpu'))
        partial = train(cfg, folder / 'resume', 'cpu', stop_after=2)
        resumed = load_checkpoint(train(cfg, folder / 'resume', 'cpu', resume=partial))
        for key in ('model', 'optimizer', 'rng', 'cursor', 'tokens_seen', 'best_nll', 'stale_evals', 'step'):
            assert_nested_equal(self, full[key], resumed[key])
        self.assertEqual(full['model_config']['architecture'], 'modern')
        legacy_config = {'vocab_size': 32, 'context_length': 32, 'n_layers': 2,
                         'n_heads': 2, 'd_model': 16, 'dropout': 0.0}
        original = GPT(ModelConfig(**legacy_config)).eval()
        restored = GPT(ModelConfig(**legacy_config)).eval()
        restored.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(restored.config.architecture, 'classic')
        self.assertIn('positions.weight', restored.state_dict())
        ids = torch.tensor([[1, 6, 7]])
        torch.testing.assert_close(original(ids), restored(ids), atol=0, rtol=0)

    def test_default_parameter_count_and_invalid_rope_config(self):
        model = GPT(ModelConfig(architecture='modern'))
        self.assertEqual(model.parameter_count(), 17308032)
        # The output projection reuses exactly the input token matrix.
        self.assertEqual(sum(name == 'tokens.weight' for name, _ in model.named_parameters()), 1)
        self.assertFalse(any('lm_head' in name for name, _ in model.named_parameters()))
        with self.assertRaisesRegex(ValueError, 'even head width'):
            ModelConfig(architecture='modern', d_model=15, n_heads=3)
        with self.assertRaisesRegex(ValueError, 'Architecture'):
            ModelConfig(architecture='unrecognized')


if __name__ == '__main__':
    unittest.main(verbosity=2)
