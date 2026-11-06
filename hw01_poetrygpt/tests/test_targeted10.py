"""Tiny CPU contracts for controlled forks; no real corpus or remote GPU."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import shutil
import unittest
from unittest.mock import patch

import torch
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel

from poetrygpt.data import read_json
from poetrygpt.model import GPT, ModelConfig, token_losses
from poetrygpt.runtime import load_checkpoint, run_lock
from scripts import train_pretrain08 as base
from scripts import train_targeted10 as targeted
from tests import test_pretrain08 as fixtures


class OverlapStreamTests(unittest.TestCase):
    def test_zero_is_bitwise_original_with_rng_and_partial_state(self):
        data = fixtures.fixture_data()
        old = base.EpochStream(data, 71)
        old.next(11, 2)
        new = targeted.OverlapEpochStream(data, 71, old.state(), overlap=0)
        rng = torch.get_rng_state().clone()
        for quota in (1, 7, 35, 3, 64):
            left, right = old.next(quota, 3), new.next(quota, 3)
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
            self.assertEqual(old.state(), new.state())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_two_epochs_same_targets_order_cursor_and_no_cross_work(self):
        data = fixtures.fixture_data()
        old = base.EpochStream(data, 71)
        new = targeted.OverlapEpochStream(data, 71, overlap=3)
        rng = torch.get_rng_state().clone()
        saw_padding = saw_extra_context = False
        for epoch in range(2):
            remaining, observed, turn = len(data.ids) - 1, [], 0
            while remaining:
                quota = min((1, 11, 4, 17)[turn % 4], remaining)
                expected, actual = old.next(quota, 3), new.next(quota, 3)
                flatten = lambda batches: [int(t) for _, labels in batches for t in labels[labels != -100]]
                self.assertEqual(flatten(expected), flatten(actual))
                self.assertEqual(len(flatten(actual)), quota)
                observed.extend(flatten(actual))
                self.assertEqual(old.state(), new.state())
                self.assertEqual(old.windows.tolist(), new.windows.tolist())
                for ids, labels in actual:
                    self.assertLessEqual(ids.shape[1], data.context_length)
                    self.assertTrue(torch.all(labels[:, 0] == -100))
                    self.assertTrue(torch.all(labels[ids == data.pad_id] == -100))
                    saw_padding |= bool((ids == data.pad_id).any())
                    for x, y in zip(ids, labels):
                        row = x[x != data.pad_id].tolist()
                        self.assertNotIn(2, row[1:-1])
                        matches = [i for i in range(len(data.ids) - len(row) + 1)
                                   if data.ids[i:i + len(row)].tolist() == row]
                        self.assertEqual(len(matches), 1)
                        begin = matches[0]
                        owners = [r for r in data.metadata if r['start'] - 1 <= begin
                                  and begin + len(row) <= r['end']]
                        self.assertEqual(len(owners), 1)
                        first = int(y.ne(-100).nonzero()[0])
                        self.assertEqual(y[first:len(row)].tolist(), row[first:])
                        self.assertLessEqual(len(row) - first, data.context_length - 1 - 3)
                        saw_extra_context |= first > 1
                remaining -= quota
                turn += 1
            self.assertEqual(Counter(observed), Counter(data.ids[1:].tolist()))
            self.assertEqual(new.record_targets.tolist(),
                             [(epoch + 1) * (r['end'] - r['start']) for r in data.metadata])
        self.assertTrue(saw_padding and saw_extra_context)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_partial_parent_fork_resume_and_validation(self):
        data = fixtures.fixture_data()
        parent = base.EpochStream(data, 71)
        parent.next(5, 2)
        new = targeted.OverlapEpochStream(data, 71, parent.state(), overlap=3)
        new.next(1, 2)
        restored = targeted.OverlapEpochStream(data, 71, new.state(), overlap=3)
        for quota in (1, 18, 47, 3):
            left, right = new.next(quota, 2), restored.next(quota, 2)
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
            self.assertEqual(new.state(), restored.state())
        for bad in (-1, True, 1.5, 7, 128):
            with self.assertRaises(ValueError):
                targeted.OverlapEpochStream(data, 71, overlap=bad)
        for bad in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                targeted.config(dict(parent='fixture.pt', max_tokens=20, context_overlap=bad))


class NoiseTests(unittest.TestCase):
    def test_protection_labels_and_isolated_rng(self):
        tokenizer = Tokenizer(WordLevel({'<pad>': 0, '<bos>': 1, '<eos>': 2,
            'word': 3, 'line\nbreak': 4, 'next': 5}, unk_token='<pad>'))
        tokenizer.add_special_tokens([AddedToken('<extra>', special=True)])
        ids = torch.tensor([[1, 3, 4, 5, 2, 0, 6]])
        labels = torch.tensor([[-100, 3, 4, 5, 2, -100, -100]])
        original = (ids.clone(), labels.clone())
        rng = torch.get_rng_state().clone()
        noise = targeted.InputNoise(tokenizer, [0, 1, 2], 1., 77)
        corrupted = noise.apply([(ids, labels)])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(torch.equal(ids, original[0]))
        self.assertTrue(torch.equal(labels, original[1]))
        self.assertIs(corrupted[0][1], labels)
        for index in (0, 2, 4, 5, 6):
            self.assertEqual(corrupted[0][0][0, index], ids[0, index])
        self.assertEqual(set(noise.pool.tolist()), {3, 5})
        self.assertEqual(noise.counts['selected'], 2)
        restored = targeted.InputNoise(tokenizer, [0, 1, 2], 1., 77, noise.state())
        self.assertTrue(torch.equal(noise.apply([(ids, labels)])[0][0],
                                    restored.apply([(ids, labels)])[0][0]))
        zero = targeted.InputNoise(tokenizer, [0, 1, 2], 0., 77)
        batch = [(ids, labels)]
        before = zero.state()['generator'].clone()
        self.assertIs(zero.apply(batch), batch)
        self.assertTrue(torch.equal(before, zero.state()['generator']))

    def test_single_causal_shift_keeps_clean_targets(self):
        labels = torch.tensor([[-100, 3, 5, 2, -100]])
        logits = torch.zeros(1, 5, 8)
        for position, target in enumerate([3, 5, 2]):
            logits[0, position, target] = 15.
        losses, mask = token_losses(logits, labels)
        self.assertEqual(mask.tolist(), [[True, True, True, False]])
        self.assertLess(float(losses.sum()), .0001)

    def test_offset_two_excludes_cross_document_and_padding_targets(self):
        ids = torch.tensor([[1, 4, 5, 2, 7, 8, 2, 0], [1, 4, 5, 6, 2, 0, 0, 0]])
        labels = torch.tensor([[-100, 4, 5, 2, 7, 8, 2, -100],
                               [-100, -100, -100, 6, 2, -100, -100, -100]])
        original = labels.clone()
        targets = targeted.future_labels(ids, labels, [0, 1, 2])
        self.assertEqual(targets.tolist(), [[5, 2, -100, 8, 2, -100], [-100, 6, 2, -100, -100, -100]])
        self.assertTrue(torch.equal(labels, original))
        # The final EOS is predicted; the first token after an intervening EOS is not.
        logits = torch.zeros(2, 6, 10)
        for row, col in targets.ne(-100).nonzero().tolist():
            logits[row, col, targets[row, col]] = 15.
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100, reduction='sum')
        self.assertLess(float(loss), .0001)


class TargetedTests(unittest.TestCase):
    assert_nested_equal = fixtures.TrainingTests.assert_nested_equal

    def setUp(self):
        fixtures.TrainingTests.setUp(self)
        # Tiny independent CPU fixtures must not contend with real series work.
        def fixture_lock(directory, kind):
            if Path(directory) == targeted.ROOT / 'runs/pretrain_series01':
                directory = self.root / 'series_lock'
            return run_lock(directory, kind)
        for module in (base, targeted):
            replacement = patch.object(module, 'run_lock', side_effect=fixture_lock)
            replacement.start()
            self.addCleanup(replacement.stop)

    def parent(self, optimizer='adamw'):
        path = self.root / ('parent_' + optimizer)
        base.train({**self.cfg, 'optimizer': optimizer}, path, 33, 'cpu')
        return path / 'last.pt'

    def pilot(self, parent, **changes):
        return dict(parent=str(parent), max_tokens=53, cpu_threads=1,
                    eval_batches=2, eval_batch_size=2, eval_interval=2,
                    save_interval=2, log_interval=100, max_active_seconds=300, **changes)

    def equal_training(self, a, b, include_noise=True):
        keys = ['model', 'optimizers', 'rng', 'stream', 'step', 'tokens_seen',
                'local_tokens_seen', 'absolute_tokens_seen', 'best_nll']
        if include_noise:
            keys.append('noise')
        for key in keys:
            with self.subTest(field=key):
                self.assert_nested_equal(a[key], b[key])

    def test_fork_preserves_parent_and_both_optimizer_roles(self):
        for optimizer in ('adamw', 'muon'):
            with self.subTest(optimizer=optimizer):
                parent = self.parent(optimizer)
                original = load_checkpoint(parent)
                cfg = self.pilot(parent)
                run = self.root / ('fork_' + optimizer)
                targeted.train(cfg, run, 'cpu', stop_after=0)
                fork = load_checkpoint(run / 'last.pt')
                for key in ('model', 'optimizers', 'rng', 'stream'):
                    self.assert_nested_equal(original[key], fork[key])
                self.assertEqual(fork['tokens_seen'], 33)
                self.assertEqual(fork['local_tokens_seen'], 0)
                self.assertEqual(fork['absolute_tokens_seen'], 33)
                model = GPT(ModelConfig(**original['model_config']))
                opts = base.optimizers(model, original['config'], torch.device('cpu'))
                all_ids = [id(p) for opt in opts.values() for group in opt.param_groups for p in group['params']]
                self.assertEqual(len(all_ids), len(set(all_ids)))
                self.assertEqual(set(all_ids), {id(p) for p in model.parameters()})
                if optimizer == 'muon':
                    self.assertEqual(set(fork['optimizers']), {'muon', 'adamw'})
                    muon_ids = {id(p) for g in opts['muon'].param_groups for p in g['params']}
                    self.assertNotIn(id(model.tokens.weight), muon_ids)
                    self.assertTrue(all(id(p) not in muon_ids for p in model.parameters() if p.ndim != 2))
                left = base.EpochStream(fixtures.fixture_data(), original['config']['seed'], original['stream'])
                right = base.EpochStream(fixtures.fixture_data(), original['config']['seed'], fork['stream'])
                for a, b in zip(left.next(11, 3), right.next(11, 3)):
                    self.assert_nested_equal(a, b)

    def test_zero_noise_exactly_matches_clean_control(self):
        for optimizer in ('adamw', 'muon'):
            with self.subTest(optimizer=optimizer):
                parent = self.parent(optimizer)
                paths = []
                for mode in ('ce', 'noise'):
                    run = self.root / (optimizer + '_' + mode)
                    targeted.train(self.pilot(parent, mode=mode, noise_probability=0.), run, 'cpu')
                    paths.append(load_checkpoint(run / 'last.pt'))
                self.equal_training(*paths)

    def test_clean_fork_matches_original_pretrain_continuation(self):
        for optimizer in ('adamw', 'muon'):
            with self.subTest(optimizer=optimizer):
                parent = self.parent(optimizer)
                run = self.root / (optimizer + '_control_equivalence')
                targeted.train(self.pilot(parent), run, 'cpu')
                pilot = load_checkpoint(run / 'last.pt')
                base.train({**self.cfg, 'optimizer': optimizer}, parent.parent, 86, 'cpu', resume=True)
                reference = load_checkpoint(parent)
                for key in ('model', 'optimizers', 'rng', 'stream'):
                    with self.subTest(field=key):
                        self.assert_nested_equal(reference[key], pilot[key])
                self.assertEqual(reference['tokens_seen'], pilot['absolute_tokens_seen'])
                self.assertEqual(reference['tokens_seen'], pilot['tokens_seen'])
                self.assertEqual(pilot['local_tokens_seen'], 53)

    def test_noise_resume_with_short_final_update_and_changed_batch(self):
        for optimizer in ('adamw', 'muon'):
            with self.subTest(optimizer=optimizer):
                parent = self.parent(optimizer)
                cfg = self.pilot(parent, mode='noise', noise_probability=.2, target_tokens=7, batch_size=2)
                full, split = self.root / (optimizer + '_full'), self.root / (optimizer + '_split')
                targeted.train(cfg, full, 'cpu')
                targeted.train(cfg, split, 'cpu', stop_after=4)
                targeted.train(cfg, split, 'cpu', resume=True)
                a, b = load_checkpoint(full / 'last.pt'), load_checkpoint(split / 'last.pt')
                self.equal_training(a, b)
                self.assertEqual(a['step'], 8)
                self.assertEqual(a['absolute_tokens_seen'], 86)
                self.assertEqual(a['tokens_seen'], 86)
                self.assertEqual(a['local_tokens_seen'], 53)
                self.assertGreater(a['noise']['counts']['selected'], 0)
                events = [json.loads(x) for x in (full / 'metrics.jsonl').read_text('utf-8').splitlines()]
                train_events = [x for x in events if x['event'] == 'train']
                self.assertEqual(train_events[-1]['supervised_targets'], 4)
                for e in train_events:
                    self.assertEqual(e['lr'], base.token_lr(e['absolute_targets'], self.cfg))

    def test_overlap_fork_exact_resume_adamw_muon_and_final_quota(self):
        for optimizer in ('adamw', 'muon'):
            with self.subTest(optimizer=optimizer):
                parent = self.parent(optimizer)
                cfg = self.pilot(parent, context_overlap=3, target_tokens=7, batch_size=2)
                full, split = self.root / (optimizer + '_overlap_full'), self.root / (optimizer + '_overlap_split')
                targeted.train(cfg, full, 'cpu')
                targeted.train(cfg, split, 'cpu', stop_after=3)
                with self.assertRaisesRegex(ValueError, 'configuration'):
                    targeted.train({**cfg, 'context_overlap': 2}, split, 'cpu', resume=True)
                targeted.train(cfg, split, 'cpu', resume=True)
                a, b = load_checkpoint(full / 'last.pt'), load_checkpoint(split / 'last.pt')
                self.equal_training(a, b)
                self.assertEqual(a['tokens_seen'], 86)
                self.assertEqual(a['local_tokens_seen'], 53)
                self.assertEqual(a['step'], 8)
                control = base.EpochStream(fixtures.fixture_data(), self.cfg['seed'], load_checkpoint(parent)['stream'])
                control.next(53, 2)
                self.assertEqual(a['stream'], control.state())
                events = [json.loads(x) for x in (full / 'metrics.jsonl').read_text('utf-8').splitlines()]
                self.assertEqual(events[0]['context_overlap'], 3)
                self.assertEqual([e for e in events if e['event'] == 'train'][-1]['supervised_targets'], 4)

    def test_stage_parent_uses_exact_run_identity_and_rejects_mismatch(self):
        parent = self.parent()
        stage = parent.parent / 'stages' / '33.pt'
        stage.parent.mkdir()
        shutil.copyfile(parent, stage)
        run = self.root / 'from_stage'
        targeted.train(self.pilot(stage), run, 'cpu', stop_after=0)
        fork = load_checkpoint(run / 'last.pt')
        self.assertEqual(fork['tokens_seen'], 33)
        self.assertEqual(fork['local_tokens_seen'], 0)
        self.assertEqual(Path(fork['parent_record']['environment_path']), parent.parent / 'environment.json')
        identity = parent.parent / 'environment.json'
        changed = read_json(identity)
        changed['config']['lr'] *= .5
        identity.write_text(json.dumps(changed), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            targeted.train(self.pilot(stage), self.root / 'bad_stage_identity', 'cpu', stop_after=0)

    def test_resume_does_not_reopen_parent_and_rejects_changes(self):
        parent = self.parent()
        cfg = self.pilot(parent)
        run = self.root / 'resume_guard'
        targeted.train(cfg, run, 'cpu', stop_after=2)
        parent.unlink()  # Only this tiny fixture; independent resume uses local full state.
        with self.assertRaisesRegex(ValueError, 'configuration'):
            targeted.train({**cfg, 'lr_multiplier': .5}, run, 'cpu', resume=True)
        targeted.train(cfg, run, 'cpu', resume=True)
        self.assertEqual(read_json(run / 'status.json')['status'], 'completed')

    def test_lr_restart_is_explicit_and_inherit_uses_absolute_tokens(self):
        parent = self.parent()
        cfg = targeted.config(self.pilot(parent))
        self.assertEqual(targeted.pilot_lr(11, 33, self.cfg, cfg), base.token_lr(44, self.cfg))
        with self.assertRaisesRegex(ValueError, 'silently'):
            targeted.config(self.pilot(parent, lr=.0001))
        restart = targeted.config(self.pilot(parent, schedule='restart', lr=.0001, min_lr=.00001, warmup_tokens=5))
        self.assertEqual(targeted.pilot_lr(5, 33, self.cfg, restart), .0001)
        self.assertAlmostEqual(targeted.pilot_lr(53, 33, self.cfg, restart), .00001)
        for bad in [dict(mode='ce', noise_probability=.05), dict(noise_probability=.3), dict(lr_multiplier=0),
                    dict(adam_beta2=0), dict(adam_beta2=1), dict(adam_beta2=float('nan')),
                    dict(adam_beta2=True), dict(weight_decay_multiplier=0), dict(weight_decay_multiplier=float('inf'))]:
            with self.assertRaises(ValueError):
                targeted.config(self.pilot(parent, **bad))

    def test_small_batch_hyperparameters_preserve_moments_and_resume(self):
        beta2 = .95 ** .5
        for optimizer in ('adamw', 'muon'):
            parent = self.parent(optimizer)
            original = load_checkpoint(parent)
            for mode in ('ce', 'mtp2'):
                with self.subTest(optimizer=optimizer, mode=mode):
                    cfg = self.pilot(parent, mode=mode, mtp_alpha=.1 if mode == 'mtp2' else 0.,
                        target_tokens=7, batch_size=2, adam_beta2=beta2, weight_decay_multiplier=.5)
                    full = self.root / (optimizer + '_' + mode + '_override_full')
                    split = self.root / (optimizer + '_' + mode + '_override_split')
                    targeted.train(cfg, split, 'cpu', stop_after=0)
                    fork = load_checkpoint(split / 'last.pt')
                    for name, saved in original['optimizers'].items():
                        self.assert_nested_equal(saved['state'], fork['optimizers'][name]['state'])
                        for old, new in zip(saved['param_groups'], fork['optimizers'][name]['param_groups']):
                            expected = deepcopy(old)
                            expected['weight_decay'] *= .5
                            if name == 'adamw':
                                expected['betas'] = (old['betas'][0], beta2)
                            self.assert_nested_equal(expected, new)
                    targeted.train(cfg, full, 'cpu')
                    targeted.train(cfg, split, 'cpu', resume=True, stop_after=4)
                    targeted.train(cfg, split, 'cpu', resume=True)
                    a, b = load_checkpoint(full / 'last.pt'), load_checkpoint(split / 'last.pt')
                    self.equal_training(a, b)
                    self.assertEqual(a['config']['weight_decay'], original['config']['weight_decay'] * .5)
                    self.assertEqual(a['config']['betas'], [original['config']['betas'][0], beta2])
                    for name, saved in original['optimizers'].items():
                        for old, new in zip(saved['param_groups'], b['optimizers'][name]['param_groups']):
                            self.assertEqual(new['weight_decay'], old['weight_decay'] * .5)
                            if name == 'adamw':
                                self.assertEqual(new['betas'], (old['betas'][0], beta2))
                            else:
                                self.assertEqual(new['momentum'], old['momentum'])
                    if mode == 'mtp2':
                        self.assert_nested_equal(a['mtp2'], b['mtp2'])
                        group = b['mtp2']['optimizer']['param_groups'][0]
                        self.assertEqual(group['weight_decay'], original['config']['weight_decay'] * .5)
                        self.assertEqual(group['betas'], (original['config']['betas'][0], beta2))
                    events = [json.loads(line) for line in (split / 'metrics.jsonl').read_text('utf-8').splitlines()]
                    for event in [e for e in events if e['event'] in ('fork', 'resume')]:
                        self.assertEqual(event['weight_decay_multiplier'], .5)
                        self.assertEqual(event['adam_beta2'], beta2)
                        for name, groups in event['optimizer_groups'].items():
                            for old, group in zip(original['optimizers'][name]['param_groups'], groups):
                                self.assertEqual(group['weight_decay'], old['weight_decay'] * .5)

    def test_mtp_initialization_ownership_rng_and_manual_objective(self):
        model = GPT(ModelConfig(**{**self.cfg['model'], 'dropout': 0.}))
        before = torch.get_rng_state().clone()
        original_keys = set(model.state_dict())
        mtp = targeted.MTP2(model, self.cfg, 'tokenizer', [0, 1, 2])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(mtp.head.weight, model.tokens.weight))
        self.assertNotEqual(mtp.head.weight.data_ptr(), model.tokens.weight.data_ptr())
        self.assertEqual(set(model.state_dict()), original_keys)
        self.assertIsInstance(mtp.optimizer, torch.optim.AdamW)
        base_opts = base.optimizers(model, {**self.cfg, 'optimizer': 'muon'}, torch.device('cpu'))
        base_ids = {id(p) for opt in base_opts.values() for g in opt.param_groups for p in g['params']}
        self.assertNotIn(id(mtp.head.weight), base_ids)
        self.assertEqual({id(p) for g in mtp.optimizer.param_groups for p in g['params']}, {id(mtp.head.weight)})
        ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 8, 2, 0]])
        labels = torch.tensor([[-100, 4, 5, 6, 2], [-100, 7, 8, 2, -100]])
        captured = []
        hook = model.final_norm.register_forward_hook(lambda m, args, result: captured.append(result))
        logits = model(ids)
        hook.remove()
        expected_main = token_losses(logits, labels)[0].sum() / 7
        # Explicit x[t+2] target pairs, without using future_labels for the oracle.
        selected_hidden = captured[0][[0, 0, 0, 1, 1], [0, 1, 2, 0, 1]]
        expected_aux = torch.nn.functional.cross_entropy(mtp.head(selected_hidden), torch.tensor([5, 6, 2, 8, 2]))
        expected = expected_main + .1 * expected_aux
        expected.backward()
        expected_base_grads = [p.grad.clone() for p in model.parameters()]
        expected_head_grad = mtp.head.weight.grad.clone()
        for p in model.parameters():
            p.grad = None
        mtp.head.weight.grad = None
        report = mtp.backward(model, [(ids[:1], labels[:1]), (ids[1:], labels[1:])],
                              7, torch.device('cpu'), 'fp32', .1)
        self.assertEqual(report['auxiliary_targets'], 5)
        self.assertAlmostEqual(report['main_nll'], float(expected_main.detach()), places=6)
        self.assertAlmostEqual(report['auxiliary_nll'], float(expected_aux.detach()), places=6)
        self.assertAlmostEqual(report['total_loss'], float(expected.detach()), places=6)
        for actual, expected_grad in zip(model.parameters(), expected_base_grads):
            torch.testing.assert_close(actual.grad, expected_grad, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(mtp.head.weight.grad, expected_head_grad, rtol=3e-5, atol=3e-6)
        self.assertFalse(model.final_norm._forward_hooks)

    def test_zero_alpha_is_exact_ce_for_adamw_and_muon(self):
        for optimizer in ('adamw', 'muon'):
            with self.subTest(optimizer=optimizer):
                parent = self.parent(optimizer)
                outputs = []
                for mode in ('ce', 'mtp2'):
                    run = self.root / (optimizer + '_zero_alpha_' + mode)
                    targeted.train(self.pilot(parent, mode=mode, mtp_alpha=0.), run, 'cpu')
                    outputs.append(load_checkpoint(run / 'last.pt'))
                self.equal_training(*outputs)
                self.assertEqual(outputs[1]['mtp2']['targets_seen'], 0)
                self.assertFalse(outputs[1]['mtp2']['optimizer']['state'])

    def test_mtp_resume_base_export_and_separate_auxiliary_accounting(self):
        for optimizer in ('adamw', 'muon'):
            with self.subTest(optimizer=optimizer):
                parent = self.parent(optimizer)
                cfg = self.pilot(parent, mode='mtp2', mtp_alpha=.1, mtp_ramp_tokens=22)
                full, split = self.root / (optimizer + '_mtp_full'), self.root / (optimizer + '_mtp_split')
                targeted.train(cfg, full, 'cpu')
                targeted.train(cfg, split, 'cpu', stop_after=2)
                targeted.train(cfg, split, 'cpu', resume=True)
                a, b = load_checkpoint(full / 'last.pt'), load_checkpoint(split / 'last.pt')
                self.equal_training(a, b)
                self.assert_nested_equal(a['mtp2'], b['mtp2'])
                self.assertEqual(a['tokens_seen'], 86)
                self.assertEqual(a['local_tokens_seen'], 53)
                self.assertGreater(a['mtp2']['targets_seen'], 0)
                exported = GPT(ModelConfig(**a['model_config']))
                exported.load_state_dict(a['model'], strict=True)
                self.assertFalse(exported.final_norm._forward_hooks)
                events = [json.loads(x) for x in (full / 'metrics.jsonl').read_text('utf-8').splitlines()]
                updates = [e for e in events if e['event'] == 'train']
                self.assertEqual(updates[0]['alpha'], .05)
                self.assertEqual(updates[-1]['alpha'], .1)
                self.assertEqual(sum(e['auxiliary_targets'] for e in updates), a['mtp2']['targets_seen'])
                self.assertEqual(sum(e['supervised_targets'] for e in updates), 53)
                for event in updates:
                    self.assertEqual(event['nll'], event['main_nll'])
                    self.assertAlmostEqual(event['total_loss'], event['main_nll'] + event['alpha'] * event['auxiliary_nll'])


if __name__ == '__main__':
    unittest.main()
