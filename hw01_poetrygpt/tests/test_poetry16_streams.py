"""CPU-only synthetic target coverage, context, masking and resume contracts."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import unittest
import uuid

import numpy as np
import torch

from poetrygpt.data import PreparedDataset, sha256
from poetrygpt.model import token_losses
from scripts.poetry16_streams import TargetRecordStream, UnitWindowStream, PoetryAuxStream
from scripts.train_pretrain08 import EpochStream
from test_poetry08_stream import fixture, SPECIAL, supervised


class StreamTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parent
        folder = test_root / ('poetry16_stream_' + uuid.uuid4().hex)
        folder.mkdir()
        self.assertEqual(folder.resolve().parent, test_root)
        self.addCleanup(shutil.rmtree, folder)
        self.records = [([1, 10, 11, 12, 13, 14, 2], [3, 5, 6]),
                        ([1, 20, 21, 22, 23], [1, 3]),
                        ([1, 30, 31, 32, 33, 34, 35, 36, 37], [4, 7])]
        ids = np.zeros((3, 12), dtype=np.int64)
        labels = np.full_like(ids, -100)
        attention = np.zeros_like(ids)
        for index, (tokens, positions) in enumerate(self.records):
            ids[index, :len(tokens)] = tokens
            attention[index, :len(tokens)] = 1
            labels[index, positions] = ids[index, positions]
        np.savez(folder/'sft_train.npz', input_ids=ids, labels=labels, attention_mask=attention)
        (folder/'sft_train_index.json').write_text(json.dumps([{'id': str(i)} for i in range(3)]), encoding='utf-8')
        manifest = dict(context_length=12, special_tokens=SPECIAL,
                        output_sha256={p.name: sha256(p) for p in folder.iterdir()})
        (folder/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        self.aux = PreparedDataset(folder, 'sft', 'train', 12)
        self.main = fixture((4, 6, 3), context=5)
        windows = []
        for owner, record in enumerate(self.main.metadata):
            start = record['start'] - 1
            while start + 1 < record['end']:
                end = min(start + self.main.context_length, record['end'])
                windows.append((start, end, owner))
                start = end - 1
        self.windows = np.asarray(windows, dtype=np.int64)

    def assert_batches_equal(self, a, b):
        self.assertEqual(len(a), len(b))
        for pair_a, pair_b in zip(a, b):
            for tensor_a, tensor_b in zip(pair_a, pair_b):
                self.assertTrue(torch.equal(tensor_a, tensor_b))

    def test_auxiliary_full_pass_preserves_exact_targets_without_eos_insertion(self):
        stream = TargetRecordStream(self.aux, 19)
        expected = [tokens[position] for tokens, positions in self.records for position in positions]
        batches = stream.next(len(expected), 2)
        self.assertEqual(Counter(supervised(batches)), Counter(expected))
        self.assertEqual(stream.record_targets.tolist(), [3, 2, 2])
        self.assertEqual(stream.total_targets, 7)
        self.assertIsNone(stream.pending)
        self.assertEqual(supervised(batches).count(2), 1)  # Only the one original supervised EOS.
        self.assertEqual(stream.statistics()['targets_per_epoch'], 7)
        next_epoch = stream.next(7, 2)
        self.assertEqual(Counter(supervised(next_epoch)), Counter(expected))
        self.assertEqual(stream.record_targets.tolist(), [6, 4, 4])

    def test_auxiliary_single_target_cuts_keep_full_prefix_and_every_mask(self):
        stream = TargetRecordStream(self.aux, 19)
        seen = []
        for _ in range(7):
            batches = stream.next(1, 3)
            self.assertEqual(len(batches), 1)
            ids, labels = batches[0]
            tokens = ids[0].tolist()
            original, allowed = next((t, p) for t, p in self.records if t[1] == tokens[1])
            self.assertEqual(tokens, original[:len(tokens)])
            positions = torch.nonzero(labels[0] != -100).flatten().tolist()
            self.assertEqual(len(positions), 1)
            self.assertIn(positions[0], allowed)
            self.assertEqual(positions[0], len(tokens)-1)
            self.assertEqual(labels[0, :positions[0]].tolist(), [-100]*positions[0])
            seen.extend(supervised(batches))
        self.assertEqual(len(seen), 7)
        self.assertEqual(Counter(seen), Counter(t[p] for t, ps in self.records for p in ps))

    def test_masked_positions_have_no_loss_and_padding_is_masked(self):
        batches = TargetRecordStream(self.aux, 19).next(7, 3)
        for ids, labels in batches:
            self.assertTrue(torch.all(labels[ids == SPECIAL['<|pad|>']] == -100))
            logits = torch.zeros((*ids.shape, 64), dtype=torch.float32)
            before, mask = token_losses(logits, labels)
            changed = logits.clone()
            unused_predictions = labels[:, 1:] == -100
            changed[:, :-1, 0][unused_predictions] = 1000.
            after, changed_mask = token_losses(changed, labels)
            self.assertTrue(torch.equal(mask, changed_mask))
            self.assertTrue(torch.equal(before, after))
            self.assertEqual(int(mask.sum()), int(labels[:, 1:].ne(-100).sum()))

    def test_auxiliary_resume_and_quota_partition_are_exact(self):
        stream = TargetRecordStream(self.aux, 19)
        stream.next(1, 2)
        self.assertIsNotNone(stream.pending)
        restored = TargetRecordStream(self.aux, 19, stream.state())
        for quota in [2, 1, 9, 3, 7, 4]:
            self.assert_batches_equal(stream.next(quota, 2), restored.next(quota, 2))
            self.assertEqual(stream.state(), restored.state())
        whole = TargetRecordStream(self.aux, 19)
        whole.next(stream.total_targets, 3)
        self.assertEqual(stream.state(), whole.state())

    def test_auxiliary_rejects_corrupt_masks_and_resume_state(self):
        stream = TargetRecordStream(self.aux, 19)
        stream.next(1, 2)
        for mutate in [lambda s: s.update(total_targets=s['total_targets']+1),
                       lambda s: s['cursor'].update(offset=99),
                       lambda s: s['cursor'].update(epoch=-1),
                       lambda s: s['pending'].update(index=99),
                       lambda s: s['pending'].update(consumed=0),
                       lambda s: s['record_targets'].__setitem__(0, -1),
                       lambda s: s.update(seed=20)]:
            state = deepcopy(stream.state())
            mutate(state)
            with self.assertRaises(ValueError):
                TargetRecordStream(self.aux, 19, state)
        changed = deepcopy(self.aux)
        changed.arrays['input_ids'][0, 1] += 1  # A masked context edit must also invalidate resume.
        with self.assertRaisesRegex(ValueError, 'identity'):
            TargetRecordStream(changed, 19, stream.state())
        for update in [lambda a: a['labels'].__setitem__((0, 0), 1),
                       lambda a: a['labels'].__setitem__((0, slice(None)), -100),
                       lambda a: a['labels'].__setitem__((0, 3), 99)]:
            changed = deepcopy(self.aux)
            update(changed.arrays)
            with self.assertRaises(ValueError):
                TargetRecordStream(changed, 19)

    def test_frozen_windows_cover_source_and_shuffle_without_mutation(self):
        original_ids = self.main.ids.copy()
        original_windows = self.windows.copy()
        stream = UnitWindowStream(self.main, 19, windows=self.windows)
        per_epoch = len(self.main.ids)-1
        first = stream.next(per_epoch, 2)
        second = stream.next(per_epoch, 2)
        expected = Counter(self.main.ids[1:].tolist())
        self.assertEqual(Counter(supervised(first)), expected)
        self.assertEqual(Counter(supervised(second)), expected)
        self.assertNotEqual(supervised(first), supervised(second))
        np.testing.assert_array_equal(stream.record_targets,
            [2*(r['end']-r['start']) for r in self.main.metadata])
        np.testing.assert_array_equal(self.main.ids, original_ids)
        np.testing.assert_array_equal(self.windows, original_windows)
        self.assertFalse(stream.windows.flags.writeable)

    def test_frozen_window_resume_pending_and_invalid_windows(self):
        stream = UnitWindowStream(self.main, 19, windows=self.windows)
        stream.next(1, 2)
        self.assertIsNotNone(stream.pending)
        resumed = UnitWindowStream(self.main, 19, stream.state(), windows=self.windows)
        for quota in [3, 17, 2, 9]:
            self.assert_batches_equal(stream.next(quota, 2), resumed.next(quota, 2))
            self.assertEqual(stream.state(), resumed.state())
        malformed = [self.windows.astype(float), self.windows[:, :2], self.windows[1:],
                     np.concatenate([self.windows, self.windows[:1]])]
        for col, value in [(0, -1), (1, len(self.main.ids)+1), (2, 99)]:
            bad = self.windows.copy()
            bad[0, col] = value
            malformed.append(bad)
        bad = self.windows.copy()
        bad[0, 1] = self.main.metadata[1]['end']  # Cross-owner and over-context.
        malformed.append(bad)
        for windows in malformed:
            with self.assertRaises(ValueError):
                UnitWindowStream(self.main, 19, windows=windows)
        with self.assertRaisesRegex(ValueError, 'identity'):
            UnitWindowStream(self.main, 19, stream.state(), windows=self.windows[::-1])
        for mutate in [lambda s: s.update(total_targets=s['total_targets']+1),
                       lambda s: s['cursor'].update(offset=0),
                       lambda s: s['record_targets'].__setitem__(0, 999)]:
            bad = deepcopy(stream.state())
            mutate(bad)
            with self.assertRaises(ValueError):
                UnitWindowStream(self.main, 19, bad, windows=self.windows)

    def mixture(self, state=None, numerator=1):
        saved = {} if state is None else state['domains']
        main = UnitWindowStream(self.main, 19, saved.get('poetry'), windows=self.windows)
        aux = TargetRecordStream(self.aux, 20, saved.get('auxiliary')) if numerator else None
        return PoetryAuxStream(main, aux, numerator=numerator, state=state)

    def test_mixture_small_integer_quotas_resume_and_pure_control(self):
        stream = self.mixture()
        for quota in [1]*13 + [2, 11, 3]:
            batches, counts = stream.next(quota, 2)
            self.assertEqual(len(supervised(batches)), quota)
            self.assertEqual(sum(counts.values()), quota)
            self.assertEqual(stream.streams['auxiliary'].total_targets, stream.total_targets//5)
        resumed = self.mixture(stream.state())
        for quota in [7, 21, 4, 1]:
            a, ca = stream.next(quota, 2)
            b, cb = resumed.next(quota, 2)
            self.assertEqual(ca, cb)
            self.assert_batches_equal(a, b)
            self.assertEqual(stream.state(), resumed.state())
        whole = self.mixture()
        whole.next(stream.total_targets, 2)
        self.assertEqual(stream.state(), whole.state())
        self.assertTrue(stream.statistics()['domains']['auxiliary']['active'])
        pure = self.mixture(numerator=0)
        batches, counts = pure.next(9, 2)
        self.assertEqual(counts, {'poetry': 9, 'auxiliary': 0})
        self.assertNotIn('auxiliary', pure.state()['domains'])
        self.assertFalse(pure.statistics()['domains']['auxiliary']['active'])
        self.assertEqual(pure.state(), self.mixture(pure.state(), numerator=0).state())

    def test_mixture_restoration_fraction_and_counts_are_checked(self):
        stream = self.mixture()
        stream.next(13, 2)
        for mutate in [lambda s: s.update(total_targets=14),
                       lambda s: s.update(auxiliary_fraction=[1, 4]),
                       lambda s: s['domains'].pop('auxiliary')]:
            state = deepcopy(stream.state())
            mutate(state)
            with self.assertRaises(ValueError):
                self.mixture(state)
        with self.assertRaisesRegex(ValueError, 'restored'):
            PoetryAuxStream(EpochStream(self.main, 19), TargetRecordStream(self.aux, 20), state=stream.state())
        with self.assertRaises(ValueError):
            PoetryAuxStream(EpochStream(self.main, 19), None)

    def test_all_samplers_leave_python_numpy_torch_rng_untouched(self):
        python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        stream = self.mixture()
        stream.next(47, 2)
        self.mixture(stream.state()).next(73, 3)
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
