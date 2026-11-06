"""Controlled plain-LM adaptation on a frozen selection of historical works.

Each new run uses fresh AdamW from its configured parent. Exact resume restores
weights, optimizer, RNG, both shuffled streams and partially consumed windows.
The coordinator owns the series deadline; this trainer enforces its run budget.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tokenizers import Tokenizer

from poetrygpt.data import BatchCursor, read_json, sha256
from poetrygpt.model import GPT, ModelConfig
from poetrygpt.runtime import (append_event, code_hashes, environment, load_checkpoint,
    repair_incomplete_jsonl_tail, resolve_device, restore_rng, rng_state, run_lock,
    save_checkpoint, seed_everything, write_json)
from poetrygpt.train import backward_microbatches, learning_rate, make_optimizer
from scripts.run_task_pilot02 import require_resources
from scripts.train_causal_series import check_gpu_idle, padded

DEFAULTS = dict(parent='runs/pretrain_series01/r08_66m_cooldown/last.pt',
    tokenizer='artifacts/tokenizers_v3/selected/tokenizer.json',
    historical='data/prepared_v7/expanded/prose_train.npy',
    historical_offsets=None, random_window_phase=False,
    selected='data/narrative_series04/selected.npy',
    selected_offsets='data/narrative_series04/selected_offsets.json',
    mode='control', selected_fraction=.5, target_tokens=8192, batch_size=16,
    max_tokens=12000000, max_wall_seconds=3600, lr=3e-5, min_lr=3e-6,
    warmup_fraction=.05, seed=20261102, weight_decay=.1, betas=[.9, .95],
    eps=1e-8, grad_clip=1., precision='bf16', fused_optimizer=True,
    cpu_threads=4, save_interval=100, log_interval=25)


def config(raw):
    if set(raw) - DEFAULTS.keys():
        raise ValueError(f'Unknown settings: {set(raw) - DEFAULTS.keys()}')
    cfg = {**deepcopy(DEFAULTS), **raw}
    if cfg['mode'] not in ('control', 'packed_mixture', 'scene_mixture', 'ending_mixture'):
        raise ValueError('Unknown narrative mode')
    for name in ('target_tokens', 'batch_size', 'max_tokens', 'cpu_threads', 'save_interval', 'log_interval'):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(name + ' must be a positive integer')
    if type(cfg['seed']) is not int or not 0 <= cfg['seed'] < 2**32 - 100:
        raise ValueError('seed must fit NumPy uint32 including stream offset')
    for name in ('selected_fraction', 'max_wall_seconds', 'lr', 'min_lr', 'warmup_fraction',
                 'weight_decay', 'eps', 'grad_clip'):
        if isinstance(cfg[name], bool) or not isinstance(cfg[name], (int, float)) or not math.isfinite(cfg[name]):
            raise ValueError(name + ' must be finite')
    if not 0 < cfg['selected_fraction'] < 1 or not 0 <= cfg['warmup_fraction'] < 1:
        raise ValueError('Invalid mixture or warmup fraction')
    if (not 0 < cfg['min_lr'] <= cfg['lr'] or cfg['weight_decay'] < 0
            or cfg['eps'] <= 0 or cfg['grad_clip'] <= 0 or not 0 < cfg['max_wall_seconds'] <= 6 * 3600):
        raise ValueError('Invalid optimizer or wall budget')
    if (not isinstance(cfg['betas'], (list, tuple)) or len(cfg['betas']) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x < 1 for x in cfg['betas'])):
        raise ValueError('Invalid AdamW betas')
    if cfg['precision'] not in ('bf16', 'fp32') or type(cfg['fused_optimizer']) is not bool:
        raise ValueError('Expected BF16/FP32 and boolean fused_optimizer')
    for name in ('parent', 'tokenizer', 'historical', 'selected', 'selected_offsets'):
        if not isinstance(cfg[name], str) or not cfg[name]:
            raise ValueError('Expected path string: ' + name)
    if cfg['historical_offsets'] is not None and (
            not isinstance(cfg['historical_offsets'], str) or not cfg['historical_offsets']):
        raise ValueError('Expected path string or null: historical_offsets')
    if type(cfg['random_window_phase']) is not bool:
        raise ValueError('Expected boolean random_window_phase')
    if cfg['random_window_phase'] and (
            cfg['historical_offsets'] is None or cfg['mode'] == 'packed_mixture'):
        raise ValueError('Random window phase requires within-work windows in every stream')
    return cfg


def scene_windows(tokens, offsets, context, eos, phase_seed=None, initial_token=None):
    """Every work target once per epoch; optional random phase is fixed per run."""
    initial_token = eos if initial_token is None else initial_token
    if context < 3 or tokens.ndim != 1 or len(tokens) < 3 or int(tokens[0]) != initial_token:
        raise ValueError('Work stream must start with its real initial token and contain work targets')
    windows, previous_end = [], 1
    for index, row in enumerate(offsets):
        start, end = row['start'], row['end']
        if (type(start) is not int or type(end) is not int or start != previous_end
                or not start < end <= len(tokens) or not row.get('family_id')
                or int(tokens[start - 1]) != (initial_token if index == 0 else eos)
                or int(tokens[end - 1]) != eos):
            raise ValueError('Selected offsets must partition whole EOS-terminated works')
        # The real preceding EOS (or initial BOS) is context only. Later windows
        # overlap one genuine token, so each next-token target survives once.
        position = start - 1
        first_targets = context - 1
        if phase_seed is not None:
            # Per-record local RNG keeps broad A/B windows identical even when
            # another stream is present, without consuming model/dropout RNG.
            generator = np.random.default_rng([phase_seed, index])
            first_targets = int(generator.integers(1, context))
        stop = min(position + first_targets + 1, end)
        windows.append((position, stop, index))
        for position in range(stop - 1, end - 1, context - 1):
            windows.append((position, min(position + context, end), index))
        previous_end = end
    if not offsets or previous_end != len(tokens):
        raise ValueError('Selected offsets do not cover the full stream')
    return windows


def window_record(tokens, start, end, ending=False):
    ids = np.asarray(tokens[start:end], dtype=np.int64).tolist()
    if len(ids) < 2:
        raise ValueError('Window has no next-token targets')
    # Ending-only changes the supervision mask, not the inputs/windows.
    prefix_targets = min(128, (len(ids) - 1) // 2) if ending else 0
    labels = [-100] * (prefix_targets + 1) + ids[prefix_targets + 1:]
    return ids, labels


class Stream:
    """Shuffled windows with retained tails and cumulative exact mixture quotas."""
    def __init__(self, cfg, tokenizer, context, state=None):
        self.cfg, self.context = cfg, context
        self.pad = tokenizer.token_to_id('<|pad|>')
        eos = tokenizer.token_to_id('<|eos|>')
        if self.pad is None or eos is None:
            raise ValueError('Missing pad/EOS tokens')
        self.historical = np.load(cfg['historical'], mmap_mode='r', allow_pickle=False)
        if (self.historical.ndim != 1 or len(self.historical) < 2
                or self.historical.dtype.kind not in 'iu'):
            raise ValueError('Expected a one-dimensional integer historical stream')
        self.arrays = {'historical': self.historical}
        self.windows = {}
        self.offsets, self.ends = [], np.array([], dtype=np.int64)
        sizes = {'historical': math.ceil((len(self.historical) - 1) / (context - 1))}
        phase_seed = cfg['seed'] if cfg['random_window_phase'] else None
        if cfg['historical_offsets'] is not None:
            historical_offsets = read_json(cfg['historical_offsets'])
            bos = tokenizer.token_to_id('<|bos|>')
            initial_token = bos if bos is not None and int(self.historical[0]) == bos else eos
            self.windows['historical'] = scene_windows(
                self.historical, historical_offsets, context, eos, phase_seed,
                initial_token=initial_token)
            sizes['historical'] = len(self.windows['historical'])
        if cfg['mode'] != 'control':
            selected = np.load(cfg['selected'], mmap_mode='r', allow_pickle=False)
            if selected.dtype.kind not in 'iu':
                raise ValueError('Expected integer selected tokens')
            self.offsets = read_json(cfg['selected_offsets'])
            scenes = scene_windows(selected, self.offsets, context, eos,
                None if phase_seed is None else phase_seed + 100)
            self.ends = np.asarray([r['end'] for r in self.offsets], dtype=np.int64)
            self.arrays['selected'] = selected
            if cfg['mode'] != 'packed_mixture':
                self.windows['selected'] = scenes
                sizes['selected'] = len(scenes)
            else:
                sizes['selected'] = math.ceil((len(selected) - 1) / (context - 1))
        self.cursors = {name: BatchCursor(size, 1, cfg['seed'] + (100 if name == 'selected' else 0))
                        for name, size in sizes.items()}
        self.pending = {name: None for name in sizes}
        self.total_targets = 0
        self.selected_targets = 0
        self.window_draws = {name: Counter() for name in sizes}
        self.family_targets = Counter()
        self.family_window_draws = Counter()
        self.input_positions = Counter()
        self.ending_context_only_targets_drawn = 0
        if state is not None:
            if set(state['cursors']) != set(self.cursors):
                raise ValueError('Resume stream domains changed')
            self.cursors = {name: BatchCursor(sizes[name], 1,
                cfg['seed'] + (100 if name == 'selected' else 0), **saved)
                for name, saved in state['cursors'].items()}
            self.pending = deepcopy(state['pending'])
            self.total_targets, self.selected_targets = state['total_targets'], state['selected_targets']
            self.window_draws = {name: Counter(values) for name, values in state['window_draws'].items()}
            self.family_targets = Counter(state['family_targets'])
            self.family_window_draws = Counter(state['family_window_draws'])
            self.input_positions = Counter(state['input_positions'])
            self.ending_context_only_targets_drawn = state['ending_context_only_targets_drawn']

    def record(self, name, index):
        if name in self.windows:
            start, end, _ = self.windows[name][index]
        else:
            start = index * (self.context - 1)
            end = min(start + self.context, len(self.arrays[name]))
        ids, labels = window_record(self.arrays[name], start, end,
            ending=name == 'selected' and self.cfg['mode'] == 'ending_mixture')
        return ids, labels, start

    def state(self):
        return dict(cursors={name: cursor.state_dict() for name, cursor in self.cursors.items()},
                    pending=deepcopy(self.pending), total_targets=self.total_targets,
                    selected_targets=self.selected_targets,
                    window_draws={name: dict(values) for name, values in self.window_draws.items()},
                    family_targets=dict(self.family_targets), family_window_draws=dict(self.family_window_draws),
                    input_positions=dict(self.input_positions),
                    ending_context_only_targets_drawn=self.ending_context_only_targets_drawn)

    def statistics(self):
        return dict(total_targets=self.total_targets, selected_targets=self.selected_targets,
                    historical_targets=self.total_targets - self.selected_targets,
                    selected_fraction=self.selected_targets / max(1, self.total_targets),
                    window_draws={name: sum(values.values()) for name, values in self.window_draws.items()},
                    unique_windows={name: len(values) for name, values in self.window_draws.items()},
                    family_targets=dict(self.family_targets), family_window_draws=dict(self.family_window_draws),
                    input_positions=dict(self.input_positions),
                    ending_context_only_targets_drawn=self.ending_context_only_targets_drawn,
                    discarded_quota_tail_targets=0,
                    cursors={name: cursor.state_dict() for name, cursor in self.cursors.items()})

    def take(self, name, quota):
        records = []
        while quota:
            new_window = self.pending[name] is None
            if new_window:
                index = self.cursors[name].next()[0]
                self.pending[name] = dict(index=index, consumed=0)
                self.window_draws[name][str(index)] += 1
            pending = self.pending[name]
            ids, labels, start = self.record(name, pending['index'])
            positions = [i for i, label in enumerate(labels) if label != -100]
            if new_window and name == 'selected':
                self.ending_context_only_targets_drawn += len(ids) - 1 - len(positions)
                owners = np.searchsorted(self.ends, np.asarray(positions) + start, side='right')
                for family in {self.offsets[int(i)]['family_id'] for i in owners}:
                    self.family_window_draws[family] += 1
            chosen = positions[pending['consumed']:pending['consumed'] + quota]
            if not chosen:
                raise ValueError('Invalid pending window: no unconsumed targets')
            # Keep original left context for a quota tail; mask already-scored
            # tokens. No target is dropped or supervised twice at step boundaries.
            stop = chosen[-1] + 1
            selected_labels = [-100] * stop
            for position in chosen:
                selected_labels[position] = labels[position]
            records.append((ids[:stop], selected_labels))
            self.input_positions[name] += stop
            if name == 'selected':
                owners = np.searchsorted(self.ends, np.asarray(chosen) + start, side='right')
                self.family_targets.update(self.offsets[int(i)]['family_id'] for i in owners)
            quota -= len(chosen)
            pending['consumed'] += len(chosen)
            if pending['consumed'] == len(positions):
                self.pending[name] = None
        return records

    def next(self, target_count):
        if type(target_count) is not int or target_count < 1:
            raise ValueError('Positive integer target count required')
        selected = (0 if self.cfg['mode'] == 'control' else
                    round((self.total_targets + target_count) * self.cfg['selected_fraction']) - self.selected_targets)
        historical = target_count - selected
        records = self.take('historical', historical)
        if selected:
            records.extend(self.take('selected', selected))
        batches = [padded(records[i:i + self.cfg['batch_size']], self.pad)
                   for i in range(0, len(records), self.cfg['batch_size'])]
        actual = sum(int(labels[:, 1:].ne(-100).sum()) for _, labels in batches)
        if actual != target_count:
            raise AssertionError('Mixture quota differs from actual supervised tokens')
        self.total_targets += actual
        self.selected_targets += selected
        return batches, dict(targets=actual, historical_targets=historical, selected_targets=selected)


def resume_accounting(run_dir, fingerprint, checkpoint):
    """Retain physical work after the saved update, without deleting its log."""
    records = [checkpoint]
    status_path = run_dir / 'status.json'
    if status_path.exists():
        records.append(read_json(status_path))
    metrics_path = run_dir / 'metrics.jsonl'
    if metrics_path.exists():
        # A killed writer may leave one incomplete final line. The normal
        # repair helper removes it only after resume validation succeeds.
        with metrics_path.open(encoding='utf-8') as source:
            records.extend(json.loads(line) for line in source if line.endswith('\n'))
    seconds, attempt = 0., 0
    for record in records:
        if record.get('fingerprint') != fingerprint:
            raise ValueError('Resume accounting fingerprint differs from checkpoint')
        elapsed = record.get('consumed_seconds')
        identifier = record.get('attempt_id')
        if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
                or type(identifier) is not int or identifier < 1):
            raise ValueError('Invalid resume wall time or attempt ID')
        seconds, attempt = max(seconds, elapsed), max(attempt, identifier)
    return seconds, attempt + 1


def train(raw, run_dir, device_name='cuda', resume=None, stop_after=None):
    cfg = config(raw)
    run_dir = Path(run_dir)
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError('stop_after must be a positive optimizer-step boundary')
    with run_lock(ROOT / 'runs/pretrain_series01', 'series'), run_lock(run_dir, 'training'):
        return train_locked(cfg, run_dir, device_name, resume, stop_after)


def train_locked(cfg, run_dir, device_name, resume, stop_after):
    attempt_start = time.monotonic()
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError('Refuse to overwrite existing run: ' + str(run_dir))
    if resume and Path(resume).resolve() != (run_dir / 'last.pt').resolve():
        raise ValueError('Resume only this run last.pt; fork another checkpoint as a new parent')
    torch.set_num_threads(cfg['cpu_threads'])
    if device_name == 'cuda':
        if cfg['precision'] != 'bf16':
            raise ValueError('CUDA series training requires BF16')
        check_gpu_idle()
        device, precision, resources = require_resources(True)
    else:
        device, precision = resolve_device(device_name, cfg['precision'])
        resources = {'cpu_contract_run': True}
    seed_everything(cfg['seed'])
    parent = load_checkpoint(cfg['parent'])
    tokenizer_sha = sha256(cfg['tokenizer'])
    if parent['tokenizer_sha256'] != tokenizer_sha:
        raise ValueError('Parent tokenizer differs')
    tokenizer = Tokenizer.from_file(cfg['tokenizer'])
    model_cfg = ModelConfig(**parent['model_config'])
    if tokenizer.get_vocab_size() != model_cfg.vocab_size:
        raise ValueError('Vocabulary size differs')
    paths = [cfg['parent'], cfg['tokenizer'], cfg['historical']]
    if cfg['historical_offsets'] is not None:
        paths.append(cfg['historical_offsets'])
    if cfg['mode'] != 'control':
        paths += [cfg['selected'], cfg['selected_offsets']]
    identity = dict(config=cfg, model_config=parent['model_config'],
        inputs_sha256={str(p): sha256(p) for p in paths}, code_sha256=code_hashes(),
        trainer_sha256=sha256(Path(__file__)),
        helper_sha256={name: sha256(ROOT / 'scripts' / name) for name in
                       ('train_causal_series.py', 'run_task_pilot02.py')},
        environment=environment(device, precision))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    parent_model_config = model_cfg.__dict__.copy()
    model = GPT(model_cfg).to(device)
    model.load_state_dict(parent['model'])
    optimizer = make_optimizer(model, cfg, device)
    lineage = deepcopy(parent['lineage'])
    lineage.setdefault('parents', []).append(dict(checkpoint_sha256=identity['inputs_sha256'][cfg['parent']],
        domain='prose', step=parent['step'], optimizer='new AdamW for controlled narrative adaptation'))
    parent_steps = parent.get('parent_steps', 0) + parent['step']
    parent_tokens = parent.get('parent_tokens_seen', 0) + parent['tokens_seen']
    step, tokens_seen, previous_seconds, half_saved = 0, 0, 0., False
    attempt_id, checkpoint_attempt_id = 1, None
    stream_state, saved_rng = None, None
    if resume:
        state = load_checkpoint(resume)
        if state.get('fingerprint') != fingerprint:
            raise ValueError('Exact resume requires identical config, inputs, code and runtime')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        step, tokens_seen = state['step'], state['tokens_seen']
        previous_seconds, attempt_id = resume_accounting(run_dir, fingerprint, state)
        checkpoint_attempt_id, half_saved = state['attempt_id'], state['half_saved']
        stream_state, saved_rng, lineage = state['stream'], state['rng'], state['lineage']
        del state
    del parent
    stream = Stream(cfg, tokenizer, model_cfg.context_length, stream_state)
    if stream.total_targets != tokens_seen:
        raise ValueError('Sampler supervision count differs from checkpoint')
    if saved_rng is not None:
        restore_rng(saved_rng)
    schedule = {**cfg, 'max_steps': math.ceil(cfg['max_tokens'] / cfg['target_tokens'])}
    schedule['warmup_steps'] = min(schedule['max_steps'] - 1,
                                 round(schedule['max_steps'] * cfg['warmup_fraction']))
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'environment.json', identity)
    metrics = run_dir / 'metrics.jsonl'
    repair_incomplete_jsonl_tail(metrics)
    def elapsed():
        return previous_seconds + time.monotonic() - attempt_start

    def event(name, **extra):
        append_event(metrics, dict(event=name, step=step, attempt_id=attempt_id,
            fingerprint=fingerprint, consumed_seconds=elapsed(), **extra))

    event('resume' if resume else 'start', resources=resources,
          checkpoint_attempt_id=checkpoint_attempt_id,
          rollback_after_step=step if resume else None)

    def status(name, **extra):
        write_json(run_dir / 'status.json', dict(status=name, step=step,
            fingerprint=fingerprint, attempt_id=attempt_id,
            supervised_tokens_seen=tokens_seen, max_tokens=cfg['max_tokens'],
            consumed_seconds=elapsed(), stream=stream.statistics(), **extra))

    def save(name='last.pt'):
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError('Refuse to save nonfinite weights')
        save_checkpoint(run_dir / name, dict(format='poetrygpt-v1', model_config=parent_model_config,
            config={**cfg, 'domain': 'prose', 'model': parent_model_config}, model=model.state_dict(),
            optimizer=optimizer.state_dict(), step=step, tokens_seen=tokens_seen,
            selected_tokens_seen=stream.selected_targets, stream=stream.state(), rng=rng_state(),
            lineage=lineage, fingerprint=fingerprint, tokenizer_sha256=tokenizer_sha,
            parent_steps=parent_steps, parent_tokens_seen=parent_tokens,
            consumed_seconds=elapsed(), half_saved=half_saved, attempt_id=attempt_id))

    model.train()
    if not resume:
        save()
    status('running')
    try:
        while tokens_seen < cfg['max_tokens']:
            if ((run_dir / 'STOP').exists() or elapsed() >= cfg['max_wall_seconds']
                    or (stop_after is not None and step >= stop_after)):
                save()
                status('stopped_at_update_boundary')
                return run_dir / 'last.pt'
            update_start = time.perf_counter()
            targets = min(cfg['target_tokens'], cfg['max_tokens'] - tokens_seen)
            batches, counts = stream.next(targets)
            lr = learning_rate(step, schedule)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            nll = backward_microbatches(model, batches, targets, device, precision)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            step += 1
            tokens_seen += targets
            seconds = time.perf_counter() - update_start
            event('train', nll=nll, lr=lr, **counts,
                total_supervised_tokens=tokens_seen, grad_norm_before_clip=float(norm),
                seconds=seconds, targets_per_second=targets / seconds)
            if not half_saved and tokens_seen >= math.ceil(cfg['max_tokens'] / 2):
                half_saved = True
                save('half.pt')
            if step % cfg['save_interval'] == 0 or tokens_seen == cfg['max_tokens']:
                save()
            if step % cfg['log_interval'] == 0 or tokens_seen == cfg['max_tokens']:
                status('running', last_train_nll=nll)
                print(f'step={step} targets={tokens_seen} nll={nll:.4f} seconds={elapsed():.1f}', flush=True)
        status('completed')
    except BaseException as error:
        # Stream and gradients may already have advanced within this update;
        # never replace the last complete optimizer checkpoint on failure.
        status('failed_or_interrupted', error=repr(error), recovery='Resume last.pt; replay unsaved updates')
        event('failure', error=repr(error))
        raise
    return run_dir / 'last.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    train(read_json(args.config), args.run_dir, args.device, args.resume, args.stop_after)


if __name__ == '__main__':
    main()
