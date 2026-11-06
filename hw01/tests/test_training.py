"""Meaningful CPU checks: causality, masks, normalization, tails and exact resume."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
import uuid
import shutil

import numpy as np
import torch

from poetrygpt.data import BatchCursor, PreparedDataset, sha256
from poetrygpt.evaluate import evaluate, slice_keys
from poetrygpt.model import GPT, ModelConfig, token_losses
from poetrygpt.runtime import load_checkpoint, seed_everything
from poetrygpt.train import backward_microbatches, learning_rate, train, validated_config


def json_file(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(folder):
    folder.mkdir(parents=True)
    tokenizer = folder / 'tokenizer.json'
    json_file(tokenizer, {'model': {'vocab': {str(i): i for i in range(32)}}})
    outputs = {}
    for split in ('train', 'validation', 'test'):
        ids = np.asarray([1] + [6, 7, 8, 9] * 9 + [2] + [10, 11, 12] * 7 + [2], dtype=np.uint16)
        for domain in ('prose', 'poetry'):
            path = folder / f'{domain}_{split}.npy'
            np.save(path, ids)
            index = folder / f'{domain}_{split}_offsets.json'
            json_file(index, [{'work_id': split + '_a', 'family_id': split + '_a', 'start': 1, 'end': 38, 'author': 'A'},
                              {'work_id': split + '_b', 'family_id': split + '_b', 'start': 38, 'end': len(ids), 'author': 'B'}])
            outputs.update({p.name: sha256(p) for p in (path, index)})
        sequence = [1, 3, 6, 7, 5, 4, 8, 9, 5, 3, 10, 5, 4, 11, 12, 5, 2]
        targets = [-100] * 6 + [8, 9, 5] + [-100] * 4 + [11, 12, 5, 2]
        token_array = np.zeros((3, 32), dtype=np.uint16)
        labels = np.full((3, 32), -100, dtype=np.int32)
        mask = np.zeros((3, 32), dtype=np.uint8)
        for i in range(3):
            token_array[i, :len(sequence)] = sequence
            labels[i, :len(sequence)] = targets
            mask[i, :len(sequence)] = 1
        path = folder / f'sft_{split}.npz'
        np.savez_compressed(path, input_ids=token_array, labels=labels, attention_mask=mask)
        index = folder / f'sft_{split}_index.json'
        json_file(index, [{'id': f'{split}_{i}', 'group_id': f'{split}_g{i}', 'positions': len(sequence)} for i in range(3)])
        outputs.update({p.name: sha256(p) for p in (path, index)})
    json_file(folder / 'manifest.json', {'context_length': 32, 'special_tokens': {'<|pad|>': 0},
              'output_sha256': outputs, 'inputs_sha256': {'artifacts/tokenizers_v3/selected/tokenizer.json': sha256(tokenizer)}})
    return {'data': str(folder), 'tokenizer': str(tokenizer), 'domain': 'prose',
            'model': {'vocab_size': 32, 'context_length': 32, 'n_layers': 2, 'n_heads': 2, 'd_model': 16, 'dropout': 0.2},
            'batch_size': 1, 'accumulation_steps': 2, 'max_steps': 4, 'warmup_steps': 1,
            'eval_interval': 2, 'eval_batches': 0, 'save_interval': 2, 'precision': 'fp32',
            'cpu_threads': 1, 'deterministic': True, 'fused_optimizer': False}


def assert_nested_equal(test, first, second):
    if torch.is_tensor(first):
        test.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_nested_equal(test, first[key], second[key])
    elif isinstance(first, (tuple, list)):
        test.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            assert_nested_equal(test, a, b)
    else:
        test.assertEqual(first, second)


def clean_fixture(folder):
    assert folder.resolve().parent == Path('audit/training_tests_tmp').resolve()
    assert len(folder.name) == 32 and all(c in '0123456789abcdef' for c in folder.name)
    shutil.rmtree(folder)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.folder = Path('audit/training_tests_tmp') / uuid.uuid4().hex
        cls.cfg = fixture(cls.folder / 'data')

    @classmethod
    def tearDownClass(cls):
        clean_fixture(cls.folder)

    def setUp(self):
        seed_everything(123)

    def model(self, dropout=0.0):
        return GPT(ModelConfig(vocab_size=32, context_length=32, n_layers=2, n_heads=2, d_model=16, dropout=dropout))

    def test_causality_and_right_padding(self):
        model = self.model().eval()
        first = torch.tensor([[1, 6, 7, 8, 9]])
        second = first.clone()
        second[0, 3:] = torch.tensor([14, 15])
        torch.testing.assert_close(model(first)[:, :3], model(second)[:, :3], atol=1e-7, rtol=0)
        torch.testing.assert_close(model(first), model(torch.cat([first, torch.zeros(1, 3, dtype=torch.long)], 1))[:, :5], atol=1e-7, rtol=0)

    def test_one_shift_and_mask_gradient(self):
        logits = torch.randn(1, 5, 32, requires_grad=True)
        labels = torch.tensor([[-100, -100, 8, -100, 2]])
        loss, mask = token_losses(logits, labels)
        expected = -logits[0, 1].log_softmax(-1)[8] - logits[0, 3].log_softmax(-1)[2]
        torch.testing.assert_close(loss.sum(), expected)
        self.assertEqual(int(mask.sum()), 2)
        loss.sum().backward()
        self.assertEqual(float(logits.grad[0, [0, 2, 4]].abs().sum()), 0.0)

    def test_gradient_checkpointing_preserves_dropout_and_gradients(self):
        first = self.model(dropout=0.2)
        second = self.model(dropout=0.2)
        second.load_state_dict(first.state_dict())
        second.config.gradient_checkpointing = True
        ids = torch.tensor([[1, 6, 7, 8, 2]])
        gradients = []
        for model in (first, second):
            seed_everything(456)
            token_losses(model(ids), ids)[0].sum().backward()
            gradients.append([p.grad.clone() for p in model.parameters()])
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)

    def test_modified_prepared_data_is_rejected(self):
        cfg = fixture(self.folder / 'corrupted_data')
        path = Path(cfg['data']) / 'prose_train.npy'
        values = np.load(path)
        values[2] = 15
        np.save(path, values)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            PreparedDataset(cfg['data'], 'prose', 'train', 32)

    def test_nullable_metadata_and_all_frozen_metadata_slices(self):
        keys = slice_keys({'verse': None, 'dates': None, 'sources': None}, 'poetry')
        self.assertIn('verse_meter:unknown', keys)
        self.assertIn('written_period:unknown_or_cross_period', keys)
        # Metadata-only regression: no model evaluation or test-text selection.
        for path in Path('data/prepared_v3').glob('*_offsets.json'):
            for row in json.loads(path.read_text(encoding='utf-8')):
                self.assertIn('overall', slice_keys(row, path.name.split('_')[0]))

    def test_accumulation_matches_combined_variable_target_batch(self):
        first, second = self.model(), self.model()
        second.load_state_dict(first.state_dict())
        ids = torch.tensor([[1, 6, 7, 8, 2], [1, 9, 10, 11, 2]])
        labels = torch.tensor([[-100, 6, 7, 8, 2], [-100, -100, -100, 11, 2]])
        backward_microbatches(first, [(ids[:1], labels[:1]), (ids[1:], labels[1:])], 6, torch.device('cpu'), 'fp32')
        losses, _ = token_losses(second(ids), labels)
        (losses.sum() / 6).backward()
        for p, q in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-7, rtol=2e-5)

    def test_dataset_tail_and_eval_targets(self):
        data = PreparedDataset(self.cfg['data'], 'prose', 'validation', 32)
        collected = np.concatenate([data.record(i)[1][1:] for i in range(len(data))])
        np.testing.assert_array_equal(collected, data.ids[1:])
        result = evaluate(self.model(), data, torch.device('cpu'), batch_size=2)
        self.assertEqual(result['metrics']['overall']['targets'], len(data.ids) - 1)
        self.assertEqual(result['coverage'], 'full')
        self.assertEqual(result['metrics']['overall']['families'], 2)
        sft = PreparedDataset(self.cfg['data'], 'sft', 'validation', 32)
        result = evaluate(self.model(), sft, torch.device('cpu'), batch_size=2)
        self.assertEqual(result['metrics']['overall']['targets'], 21)
        for i in range(3):
            self.assertEqual(int((sft.record(i)[1] != -100).sum()), 7)

    def test_shuffle_tail_no_loss_and_cursor_resume(self):
        cursor = BatchCursor(5, 2, 20)
        a = cursor.next()
        saved = cursor.state_dict()
        restored = BatchCursor(5, 2, 20, **saved)
        b, c = cursor.next(), cursor.next()
        self.assertEqual(b, restored.next())
        self.assertEqual(c, restored.next())
        self.assertEqual(sorted(a + b + c), list(range(5)))
        self.assertEqual(len(c), 1)

    def test_exact_resume_including_dropout_optimizer_rng(self):
        full = load_checkpoint(train(self.cfg, self.folder / 'full', 'cpu'))
        partial_path = train(self.cfg, self.folder / 'resumed', 'cpu', stop_after=2)
        resumed = load_checkpoint(train(self.cfg, self.folder / 'resumed', 'cpu', resume=partial_path))
        for key in ('model', 'optimizer', 'rng', 'cursor', 'tokens_seen', 'best_nll', 'stale_evals', 'step'):
            assert_nested_equal(self, full[key], resumed[key])
        self.assertEqual(full['lineage']['origin'], 'random_initialization')
        cfg = {**self.cfg, 'lr': 0.002}
        with self.assertRaisesRegex(ValueError, 'Exact resume'):
            train(cfg, self.folder / 'resumed', 'cpu', resume=partial_path)

    def test_new_sft_stage_and_no_external_pretraining_path(self):
        path = train(self.cfg, self.folder / 'parent', 'cpu', stop_after=1)
        cfg = {**self.cfg, 'domain': 'sft'}
        with self.assertRaisesRegex(ValueError, 'own pretrained checkpoint'):
            train(cfg, self.folder / 'invalid_sft', 'cpu')
        state = load_checkpoint(train(cfg, self.folder / 'sft', 'cpu', init_from=path, stop_after=1))
        self.assertEqual(state['step'], 1)
        self.assertEqual(len(state['lineage']['parents']), 1)
        self.assertEqual(state['tokens_seen'], 14)

    def test_schedule_and_overfit_small_pattern(self):
        cfg = validated_config(self.cfg)
        self.assertGreater(learning_rate(0, cfg), 0)
        self.assertAlmostEqual(learning_rate(cfg['max_steps'] - 1, cfg), cfg['min_lr'])
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
        ids = torch.tensor([[1, 6, 7, 8, 6, 7, 8, 2]])
        labels = ids.clone()
        labels[:, 0] = -100
        initial = token_losses(model(ids), labels)[0].mean().item()
        for _ in range(45):
            optimizer.zero_grad()
            loss = token_losses(model(ids), labels)[0].mean()
            loss.backward()
            optimizer.step()
        self.assertLess(token_losses(model(ids), labels)[0].mean().item(), initial * 0.25)


if __name__ == '__main__':
    unittest.main(verbosity=2)
