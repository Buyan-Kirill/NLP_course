"""Coverage, causal masks, optimizer coverage and exact interrupted training."""
from collections import Counter
import copy
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
import uuid

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from poetrygpt.data import read_json, sha256
from poetrygpt.model import GPT, ModelConfig, token_losses
from poetrygpt.runtime import load_checkpoint, write_json
from poetrygpt.train import backward_microbatches
from scripts.train_causal_series import padded
from scripts.train_pretrain08 import EpochStream, config, optimizers, token_lr, train


def fixture_data(context=8):
    ids, metadata = [1], []
    for owner, length in enumerate((13, 17, 21)):
        start = len(ids)
        ids.extend(range(4 + owner * 25, 4 + owner * 25 + length))
        ids.append(2)
        metadata.append(dict(start=start, end=len(ids), family_id=f'f{owner}', work_id=f'w{owner}'))
    return SimpleNamespace(ids=np.asarray(ids, dtype=np.uint16), metadata=metadata,
        context_length=context, pad_id=0,
        manifest={'special_tokens': {'<|pad|>': 0, '<|bos|>': 1, '<|eos|>': 2}})


class StreamTests(unittest.TestCase):
    def test_two_epochs_full_coverage_and_changing_cuts(self):
        data = fixture_data()
        stream = EpochStream(data, 71)
        old_windows = stream.windows.copy()
        for epoch in range(2):
            batches = stream.next(len(data.ids) - 1, 3)
            supervised = [x for _, labels in batches for x in labels[labels != -100].tolist()]
            self.assertEqual(Counter(supervised), Counter(data.ids[1:].tolist()))
            for start, end, owner in stream.windows:
                self.assertGreaterEqual(start, data.metadata[owner]['start'] - 1)
                self.assertLessEqual(end, data.metadata[owner]['end'])
            self.assertEqual(stream.total_targets, (epoch + 1) * (len(data.ids) - 1))
        self.assertNotEqual(old_windows.tolist(), stream.windows.tolist())
        np.testing.assert_array_equal(stream.record_targets,
            [2 * (r['end'] - r['start']) for r in data.metadata])

    def test_partial_window_resume_across_epoch(self):
        data = fixture_data()
        stream = EpochStream(data, 71)
        stream.next(11, 2)
        restored = EpochStream(data, 71, stream.state())
        for _ in range(20):
            first, second = stream.next(7, 2), restored.next(7, 2)
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                for x, y in zip(a, b):
                    self.assertTrue(torch.equal(x, y))
            self.assertEqual(stream.state(), restored.state())
        self.assertGreater(stream.cursor.epoch, 1)

    def test_sampler_does_not_consume_global_rng(self):
        torch.manual_seed(71)
        before = torch.get_rng_state().clone()
        stream = EpochStream(fixture_data(), 71)
        stream.next(150, 3)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        test_root = Path(__file__).resolve().parent
        self.root = test_root / ('pretrain08_fixture_' + uuid.uuid4().hex)
        self.root.mkdir()
        self.assertEqual(self.root.resolve().parent, test_root)
        self.addCleanup(shutil.rmtree, self.root)
        data = fixture_data()
        tokenizer = Tokenizer(WordLevel({str(i): i for i in range(96)}, unk_token='0'))
        self.tokenizer = self.root / 'tokenizer.json'
        tokenizer.save(str(self.tokenizer))
        for split in ('train', 'validation'):
            np.save(self.root / f'prose_{split}.npy', data.ids)
            write_json(self.root / f'prose_{split}_offsets.json', data.metadata)
        manifest = dict(context_length=8, special_tokens=data.manifest['special_tokens'],
            inputs_sha256={str(self.tokenizer): sha256(self.tokenizer)},
            output_sha256={p.name: sha256(p) for p in self.root.glob('prose_*')})
        write_json(self.root / 'manifest.json', manifest)
        self.cfg = config(dict(data=str(self.root), tokenizer=str(self.tokenizer),
            model=dict(vocab_size=96, context_length=8, n_layers=2, n_heads=2, d_model=16,
                       architecture='modern', ffn_hidden_size=32, dropout=.15),
            target_tokens=11, batch_size=3, schedule_tokens=220, warmup_tokens=22,
            precision='fp32', cpu_threads=1, fused_optimizer=False, save_interval=9,
            eval_interval=7, eval_batch_size=2, eval_batches=2, log_interval=100,
            max_active_seconds=300))

    def assert_nested_equal(self, first, second):
        if torch.is_tensor(first):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_20_vs_10_resume_10_adamw_and_muon(self):
        for optimizer in ('adamw', 'muon'):
            with self.subTest(optimizer=optimizer):
                cfg = {**self.cfg, 'optimizer': optimizer}
                full, split = self.root / (optimizer + '_full'), self.root / (optimizer + '_split')
                train(cfg, full, 220, 'cpu')
                train(cfg, split, 220, 'cpu', stop_after=10)
                train(cfg, split, 220, 'cpu', resume=True)
                a, b = load_checkpoint(full / 'last.pt'), load_checkpoint(split / 'last.pt')
                self.assertEqual(a['step'], 20)
                self.assertGreater(a['stream']['cursor']['epoch'], 1)
                for key in ('model', 'optimizers', 'rng', 'stream', 'step', 'tokens_seen', 'best_nll'):
                    self.assert_nested_equal(a[key], b[key])

    def test_optimizer_coverage_and_token_schedule(self):
        model = GPT(ModelConfig(**self.cfg['model']))
        opts = optimizers(model, {**self.cfg, 'optimizer': 'muon'}, torch.device('cpu'))
        identifiers = [id(p) for opt in opts.values() for group in opt.param_groups for p in group['params']]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(set(identifiers), {id(p) for p in model.parameters()})
        self.assertEqual(token_lr(22, self.cfg), self.cfg['lr'])
        self.assertEqual(token_lr(220, self.cfg), self.cfg['min_lr'])

    def test_three_epochs_short_final_update_exact_resume(self):
        target = 3 * (len(fixture_data().ids) - 1)
        self.assertNotEqual(target % self.cfg['target_tokens'], 0)
        for optimizer in ('adamw', 'muon'):
            cfg = {**self.cfg, 'optimizer': optimizer}
            full, split = self.root / (optimizer + '_three_full'), self.root / (optimizer + '_three_split')
            train(cfg, full, target, 'cpu')
            train(cfg, split, target, 'cpu', stop_after=7)
            train(cfg, split, target, 'cpu', resume=True)
            a, b = load_checkpoint(full / 'last.pt'), load_checkpoint(split / 'last.pt')
            self.assertEqual(a['tokens_seen'], target)
            for key in ('model', 'optimizers', 'rng', 'stream', 'step', 'tokens_seen', 'best_nll'):
                self.assert_nested_equal(a[key], b[key])

    def test_causality_padding_and_accumulation(self):
        torch.manual_seed(7)
        model = GPT(ModelConfig(**{**self.cfg['model'], 'dropout': 0.}))
        other = copy.deepcopy(model)
        records = [([1, 8, 9, 2], [-100, 8, 9, 2]), ([1, 10, 2], [-100, 10, 2])]
        batch = padded(records, 0)
        losses, mask = token_losses(model(batch[0]), batch[1])
        self.assertEqual(int(mask.sum()), 5)
        self.assertEqual(float(losses[1, -1].detach()), 0.)
        a = backward_microbatches(model, [batch], 5, torch.device('cpu'), 'fp32')
        b = backward_microbatches(other, [padded([r], 0) for r in records], 5, torch.device('cpu'), 'fp32')
        self.assertAlmostEqual(a, b, places=6)
        for x, y in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(x.grad, y.grad, rtol=2e-5, atol=2e-6)
        x = torch.tensor([[1, 8, 9, 2]])
        y = torch.tensor([[1, 8, 13, 14]])
        torch.testing.assert_close(model(x)[:, :2], model(y)[:, :2], rtol=0, atol=0)

    def test_resume_rejects_changed_config(self):
        run = self.root / 'reject'
        train(self.cfg, run, 220, 'cpu', stop_after=2)
        with self.assertRaisesRegex(ValueError, 'Exact resume'):
            train({**self.cfg, 'lr': .0004}, run, 220, 'cpu', resume=True)


if __name__ == '__main__':
    unittest.main()
