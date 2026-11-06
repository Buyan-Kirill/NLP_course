"""Small CPU-only quota, epoch, data-contract and pending-tail checks.

These tests do not train a model, open real corpora/checkpoints or use CUDA.
Full optimizer/RNG training-resume integration is checked separately.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from poetrygpt.data import read_json, sha256
from poetrygpt.model import ModelConfig
from poetrygpt.runtime import write_json
from scripts.train_poetry08 import (MixtureStream, capture_rng, config,
    load_datasets, train, update_quota, validate_checkpoint, validate_input_freeze)


SPECIAL = {'<|pad|>': 0, '<|bos|>': 1, '<|eos|>': 2,
           '<|user|>': 3, '<|assistant|>': 4, '<|eot|>': 5}


def fixture(lengths=(13, 17, 21), context=8):
    ids, metadata = [SPECIAL['<|bos|>']], []
    for owner, length in enumerate(lengths):
        start = len(ids)
        ids.extend(range(6 + owner * 25, 6 + owner * 25 + length))
        ids.append(SPECIAL['<|eos|>'])
        metadata.append(dict(start=start, end=len(ids), family_id=f'f{owner}', work_id=f'w{owner}'))
    return SimpleNamespace(ids=np.asarray(ids, dtype=np.uint16), metadata=metadata,
        context_length=context, pad_id=0, manifest={'special_tokens': SPECIAL})


def supervised(batches):
    return [value for _, labels in batches for value in labels[:, 1:][labels[:, 1:] != -100].tolist()]


class MixtureTests(unittest.TestCase):
    def test_integer_quota_does_not_lose_small_batch_remainders(self):
        datasets = {'poetry': fixture(), 'prose': fixture((4, 7))}
        stream = MixtureStream(datasets, 71, 1)
        total = 0
        for quota in [1] * 23 + [7, 19, 3, 41, 2, 11]:
            batches, counts = stream.next(quota, 3)
            total += quota
            self.assertEqual(len(supervised(batches)), quota)
            self.assertEqual(sum(counts.values()), quota)
            self.assertEqual(stream.streams['prose'].total_targets, total // 10)
            self.assertEqual(stream.streams['poetry'].total_targets, total - total // 10)
        once = MixtureStream(datasets, 71, 1)
        once.next(total, 3)
        self.assertEqual(stream.state(), once.state())

    def test_pending_tails_and_different_epochs_survive_resume(self):
        datasets = {'poetry': fixture(), 'prose': fixture((4, 7))}
        stream = MixtureStream(datasets, 71, 1)
        for _ in range(100):
            stream.next(1, 2)
            if all(s.pending is not None for s in stream.streams.values()):
                break
        self.assertTrue(all(s.pending is not None for s in stream.streams.values()))
        state = deepcopy(stream.state())
        restored = MixtureStream(datasets, 71, 1, state=state)
        for quota in [7, 11, 3, 29] * 15:
            a, counts_a = stream.next(quota, 2)
            b, counts_b = restored.next(quota, 2)
            self.assertEqual(counts_a, counts_b)
            self.assertEqual(len(a), len(b))
            for pair_a, pair_b in zip(a, b):
                for first, second in zip(pair_a, pair_b):
                    self.assertTrue(torch.equal(first, second))
            self.assertEqual(stream.state(), restored.state())
        self.assertGreater(stream.streams['poetry'].cursor.epoch, 1)
        self.assertGreater(stream.streams['prose'].cursor.epoch, 1)
        self.assertNotEqual(stream.streams['poetry'].cursor.epoch, stream.streams['prose'].cursor.epoch)

    def test_full_domain_coverage_and_within_work_boundaries(self):
        datasets = {'poetry': fixture((4, 6), 5), 'prose': fixture((2, 3), 5)}
        stream = MixtureStream(datasets, 71, 1)
        # 840 total => 756 poetry = 63 epochs, 84 prose = 12 epochs.
        batches, counts = stream.next(840, 3)
        self.assertEqual(counts, {'poetry': 756, 'prose': 84})
        expected = Counter(datasets['poetry'].ids[1:].tolist())
        expected = Counter({key: value * 63 for key, value in expected.items()})
        expected.update({key: value * 12 for key, value in Counter(datasets['prose'].ids[1:].tolist()).items()})
        self.assertEqual(Counter(supervised(batches)), expected)
        for domain, epochs in [('poetry', 63), ('prose', 12)]:
            current = stream.streams[domain]
            np.testing.assert_array_equal(current.record_targets,
                [epochs * (r['end'] - r['start']) for r in datasets[domain].metadata])
            for start, end, owner in current.windows:
                self.assertGreaterEqual(start, datasets[domain].metadata[owner]['start'] - 1)
                self.assertLessEqual(end, datasets[domain].metadata[owner]['end'])
            self.assertEqual(stream.statistics()['domains'][domain]['exposures'], epochs)

    def test_pure_and_exact_short_final_quota(self):
        data = fixture()
        stream = MixtureStream({'poetry': data}, 71, 0)
        counts = [stream.next(n, 2)[1] for n in [16, 16, 5]]
        self.assertEqual(counts, [{'poetry': n, 'prose': 0} for n in [16, 16, 5]])
        self.assertEqual(stream.total_targets, 37)
        self.assertNotIn('prose', stream.state()['domains'])
        self.assertEqual(stream.statistics()['domains']['prose']['targets'], 0)
        before = torch.get_rng_state().clone()
        restored = MixtureStream({'poetry': data}, 71, 0, state=stream.state())
        restored.next(100, 2)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_changed_fraction_and_corrupt_totals_rejected(self):
        data = {'poetry': fixture(), 'prose': fixture()}
        stream = MixtureStream(data, 71, 1)
        stream.next(13, 2)
        with self.assertRaisesRegex(ValueError, 'mixture/domains'):
            MixtureStream({'poetry': data['poetry']}, 71, 0, state=stream.state())
        state = stream.state()
        state['total_targets'] += 1
        with self.assertRaisesRegex(ValueError, 'counts'):
            MixtureStream(data, 71, 1, state=state)


class DataContractTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parent
        self.root = test_root / ('poetry08_fixture_' + uuid.uuid4().hex)
        self.root.mkdir()
        self.assertEqual(self.root.resolve().parent, test_root)
        self.addCleanup(shutil.rmtree, self.root)
        tokenizer_path = self.root / 'tokenizer.json'
        vocabulary = {**SPECIAL, **{str(i): i for i in range(6, 96)}}
        Tokenizer(WordLevel(vocabulary, unk_token='<|pad|>')).save(str(tokenizer_path))
        self.tokenizer_sha = sha256(tokenizer_path)
        self.freeze = dict(status='inputs_verified_not_authorization_to_train',
                           tokenizer_sha256=self.tokenizer_sha, datasets={})
        for domain in ('poetry', 'prose'):
            folder = self.root / domain
            folder.mkdir()
            data = fixture()
            # Intentionally no prose train files: pure must not require them.
            for split in (('train', 'validation') if domain == 'poetry' else ('validation',)):
                np.save(folder / f'{domain}_{split}.npy', data.ids)
                write_json(folder / f'{domain}_{split}_offsets.json', data.metadata)
            manifest = dict(context_length=8, special_tokens=SPECIAL,
                inputs_sha256={str(tokenizer_path): self.tokenizer_sha},
                output_sha256={p.name: sha256(p) for p in folder.iterdir()})
            if domain == 'prose':
                # Freeze/manifest retain train identity without requiring those
                # files to be mounted for the pure route's retention evaluation.
                manifest['output_sha256'].update({f'prose_train{suffix}': '0' * 64
                                                  for suffix in ('.npy', '_offsets.json')})
            write_json(folder / 'manifest.json', manifest)
            self.freeze['datasets'][domain] = dict(path=str(folder), context_length=8,
                special_tokens=SPECIAL, manifest_sha256=sha256(folder / 'manifest.json'),
                splits={split: {'files_sha256': {name: digest for name, digest in manifest['output_sha256'].items()
                                                if name.startswith(f'{domain}_{split}')}}
                        for split in ('train', 'validation')})
        self.freeze_path = self.root / 'data_freeze.json'
        write_json(self.freeze_path, self.freeze)
        self.cfg = config(dict(parent=str(self.root / 'parent.pt'),
            poetry_data=str(self.root / 'poetry'), prose_data=str(self.root / 'prose'),
            tokenizer=str(tokenizer_path), input_freeze=str(self.freeze_path),
            schedule_tokens=200, warmup_tokens=20, precision='fp32',
            model=ModelConfig(vocab_size=96, context_length=8, n_layers=2, n_heads=2,
                d_model=16, architecture='modern', ffn_hidden_size=32).__dict__))

    def parent(self):
        return dict(format='poetrygpt-v1', model_config=deepcopy(self.cfg['model']),
            tokenizer_sha256=self.tokenizer_sha, lineage={'origin': 'random_initialization', 'parents': []},
            config={'domain': 'prose'}, optimizers={'adamw': {}})

    def test_pure_reads_both_validation_sets_but_no_prose_train(self):
        training, validation = load_datasets(self.cfg, self.tokenizer_sha)
        self.assertEqual(set(training), {'poetry'})
        self.assertEqual(set(validation), {'poetry', 'prose'})
        self.assertEqual(validate_input_freeze(self.cfg, self.tokenizer_sha, training, validation),
                         sha256(self.freeze_path))
        with self.assertRaises(FileNotFoundError):
            load_datasets({**self.cfg, 'mode': '90_10'}, self.tokenizer_sha)

    def test_reject_incompatible_tokenizer_and_architecture(self):
        validate_checkpoint(self.parent(), self.cfg, self.tokenizer_sha, initial=True)
        with self.assertRaisesRegex(ValueError, 'tokenizer'):
            validate_checkpoint(self.parent(), self.cfg, 'another-tokenizer', initial=True)
        parent = self.parent()
        parent['model_config']['architecture'] = 'classic'
        with self.assertRaisesRegex(ValueError, 'architecture'):
            validate_checkpoint(parent, self.cfg, self.tokenizer_sha, initial=True)
        with self.assertRaisesRegex(ValueError, 'tokenizer'):
            load_datasets(self.cfg, 'another-tokenizer')

    def test_cpu_rng_capture_never_accesses_cuda(self):
        with patch('torch.cuda.get_rng_state_all', side_effect=AssertionError('CPU touched CUDA')):
            self.assertEqual(capture_rng(torch.device('cpu'))['cuda'], [])

    def test_reject_self_consistent_replacement_of_data_and_manifest(self):
        folder = self.root / 'poetry'
        path = folder / 'poetry_train.npy'
        ids = np.load(path, allow_pickle=False)
        ids[1] = 7
        np.save(path, ids)
        manifest = read_json(folder / 'manifest.json')
        manifest['output_sha256'][path.name] = sha256(path)
        write_json(folder / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'freeze manifest mismatch'):
            load_datasets(self.cfg, self.tokenizer_sha)

    def test_freeze_status_tokenizer_paths_and_contract_checked(self):
        mutations = [
            ('status', lambda f: f.update(status='not_checked')),
            ('tokenizer', lambda f: f.update(tokenizer_sha256='0' * 64)),
            ('path', lambda f: f['datasets']['poetry'].update(path=str(self.root / 'another'))),
            ('context', lambda f: f['datasets']['poetry'].update(context_length=9)),
            ('special', lambda f: f['datasets']['poetry'].update(special_tokens={**SPECIAL, '<|bos|>': 99})),
            ('files', lambda f: f['datasets']['poetry']['splits']['train']['files_sha256'].update(
                {'poetry_train.npy': '0' * 64})),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                changed = deepcopy(self.freeze)
                mutate(changed)
                write_json(self.freeze_path, changed)
                with self.assertRaisesRegex(ValueError, '[Ii]nput freeze'):
                    load_datasets(self.cfg, self.tokenizer_sha)
        write_json(self.freeze_path, self.freeze)

    def test_frozen_file_hashes_checked_against_loaded_files(self):
        training, validation = load_datasets(self.cfg, self.tokenizer_sha)
        training['poetry'].hashes['poetry_train.npy'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Loaded files differ'):
            validate_input_freeze(self.cfg, self.tokenizer_sha, training, validation)

    def test_declared_milestones_fix_nonmultiple_update_boundaries(self):
        cfg = config({**self.cfg, 'schedule_tokens': 61, 'target_tokens': 16,
                      'milestones': [61, 23, 23]})
        self.assertEqual(cfg['milestones'], [23, 61])
        tokens, quotas = 0, []
        while tokens < cfg['schedule_tokens']:
            quota = update_quota(tokens, cfg)
            quotas.append(quota)
            tokens += quota
        self.assertEqual(quotas, [16, 7, 16, 16, 6])
        self.assertEqual(update_quota(23, cfg), 16)
        self.assertEqual(self.cfg['milestones'], [200])
        with self.assertRaisesRegex(ValueError, 'declared milestone'):
            train(cfg, self.root / 'never_created', 24, device_name='cpu')
        self.assertFalse((self.root / 'never_created').exists())
        for invalid in ([], [0, 200], [True, 200], [23], [201], '200'):
            with self.subTest(milestones=invalid), self.assertRaisesRegex(ValueError, 'Milestones'):
                config({**self.cfg, 'milestones': invalid})


if __name__ == '__main__':
    unittest.main()
