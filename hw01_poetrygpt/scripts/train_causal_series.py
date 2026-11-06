"""Controlled causal-data adaptation. Global experiment budget belongs to coordinator.

All hypotheses initialize a NEW AdamW optimizer from the same parent weights.
Resume, unlike initialization, restores optimizer, sampling, RNG and schedule.
Historical targets and synthetic targets are counted exactly; pairwise targets
are additional compute, never additional positive language-model supervision.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poetrygpt.data import BatchCursor, read_json, sha256
from poetrygpt.model import GPT, ModelConfig, token_losses
from poetrygpt.runtime import (append_event, autocast, code_hashes, environment,
    load_checkpoint, repair_incomplete_jsonl_tail, resolve_device, restore_rng,
    rng_state, run_lock, save_checkpoint, seed_everything, write_json)
from poetrygpt.train import backward_microbatches, learning_rate, make_optimizer


DEFAULTS = dict(parent='runs/pretrain_series01/r08_66m_cooldown/last.pt',
    tokenizer='artifacts/tokenizers_v3/selected/tokenizer.json',
    historical='data/prepared_v7/expanded/prose_train.npy',
    scenes='data/causal_scenes01/train.jsonl', mode='mixed', max_tokens=12000000,
    target_tokens=8192, batch_size=16, lr=3e-5, min_lr=3e-6,
    warmup_fraction=.05, seed=20261102, pair_weight=.1, negatives=1,
    synthetic_fraction=.2, skill_weights=None, max_wall_seconds=3600,
    weight_decay=.1, betas=[.9, .95], eps=1e-8, grad_clip=1.,
    precision='bf16', fused_optimizer=True, cpu_threads=4,
    save_interval=200, log_interval=100, pair_batch_scenes=4, pair_only_scenes=18,
    pair_grouped=False, sampling_manifest=None, ending_ce_weight=0.)


def config(raw):
    if set(raw) - DEFAULTS.keys():
        raise ValueError(f'Unknown settings: {set(raw) - DEFAULTS.keys()}')
    cfg = {**deepcopy(DEFAULTS), **raw}
    if cfg['mode'] not in ('control', 'mixed', 'pair', 'pair_only') or cfg['negatives'] not in (1, 3):
        raise ValueError('Unknown experiment mode or number of negatives')
    for key in ('max_tokens', 'target_tokens', 'batch_size', 'save_interval',
                'log_interval', 'pair_batch_scenes', 'pair_only_scenes', 'cpu_threads'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(cfg['pair_grouped']) is not bool:
        raise ValueError('pair_grouped must be boolean')
    if cfg['pair_grouped'] and (cfg['mode'] != 'pair_only' or cfg['pair_only_scenes'] % 2):
        raise ValueError('pair_grouped requires pair_only mode and an even pair_only_scenes')
    if cfg['pair_grouped'] and cfg['negatives'] != 3:
        raise ValueError('pair_grouped requires all three negatives so both opposite answers are scored')
    if cfg['sampling_manifest'] is not None and (
            not isinstance(cfg['sampling_manifest'], str) or not cfg['sampling_manifest']
            or not cfg['pair_grouped'] or cfg['skill_weights'] is not None):
        raise ValueError('sampling_manifest requires grouped pairs and excludes skill_weights')
    for key in ('lr', 'min_lr', 'warmup_fraction', 'pair_weight', 'synthetic_fraction',
                'max_wall_seconds', 'weight_decay', 'eps', 'grad_clip', 'ending_ce_weight'):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]):
            raise ValueError(f'{key} must be a finite number')
    if (len(cfg['betas']) != 2 or any(not math.isfinite(x) or not 0 <= x < 1 for x in cfg['betas'])
            or cfg['weight_decay'] < 0 or cfg['eps'] <= 0 or cfg['grad_clip'] <= 0):
        raise ValueError('Invalid AdamW settings')
    if not 0 < cfg['synthetic_fraction'] < 1 or not 0 <= cfg['warmup_fraction'] < 1:
        raise ValueError('Invalid mixture or warmup fraction')
    if not 0 < cfg['min_lr'] <= cfg['lr'] or not 0 <= cfg['pair_weight'] <= 1:
        raise ValueError('Invalid LR or pair weight')
    if not 0 <= cfg['ending_ce_weight'] <= 1 or (cfg['ending_ce_weight'] and cfg['mode'] != 'pair_only'):
        raise ValueError('ending_ce_weight must be in [0, 1] and nonzero only in pair_only')
    if not 0 < cfg['max_wall_seconds'] <= 7 * 3600:
        raise ValueError('Per-run wall budget must fit series budget')
    return cfg


def padded(records, pad_id):
    width = max(len(ids) for ids, _ in records)
    ids = torch.full((len(records), width), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for i, (tokens, targets) in enumerate(records):
        ids[i, :len(tokens)] = torch.tensor(tokens)
        labels[i, :len(targets)] = torch.tensor(targets)
    return ids, labels


def candidate_record(tokenizer, prefix, ending, context):
    """Mask prompt and padding, with exactly one causal shift in token_losses."""
    prompt = tokenizer.encode(prefix, add_special_tokens=False).ids
    ids = tokenizer.encode(prefix + ending, add_special_tokens=False).ids
    if not prompt or ids[:len(prompt)] != prompt or len(ids) <= len(prompt):
        raise ValueError('Unstable prompt/candidate BPE boundary or empty candidate')
    eos = tokenizer.token_to_id('<|eos|>')
    if eos is None:
        raise ValueError('Candidate scoring requires declared EOS')
    # Match evaluation and the EOS boundary at the start of packed LM scenes.
    ids = [eos] + ids
    if len(ids) > context:
        raise ValueError('Candidate exceeds context; silent truncation is forbidden')
    boundary = len(prompt) + 1
    return ids, [-100] * boundary + ids[boundary:]


def continuation_scores(logits, labels):
    losses, mask = token_losses(logits, labels)
    counts = mask.sum(1)
    if (counts == 0).any():
        raise ValueError('Empty candidate target')
    return -losses.sum(1) / counts


def pair_objective(scores):
    """scores[scene, candidate]; column zero is positive. Mean over negatives."""
    return F.softplus(scores[:, 1:] - scores[:, :1]).mean(1)


class Stream:
    """Independent deterministic LM stream; synthetic windows preserve all tails."""
    def __init__(self, cfg, tokenizer, context, state=None):
        self.cfg, self.context = cfg, context
        self.old = np.load(cfg['historical'], mmap_mode='r', allow_pickle=False)
        if self.old.ndim != 1 or len(self.old) < 2:
            raise ValueError('Expected one-dimensional historical token stream')
        self.pad = tokenizer.token_to_id('<|pad|>')
        self.eos = tokenizer.token_to_id('<|eos|>')
        if self.pad is None or self.eos is None:
            raise ValueError('Tokenizer must declare pad and EOS')
        self.rows = []
        pair_rows = {}
        if cfg['mode'] != 'control':
            for line in Path(cfg['scenes']).read_text(encoding='utf-8-sig').splitlines():
                row = json.loads(line)
                tokens = tokenizer.encode(row['text'], add_special_tokens=False).ids + [self.eos]
                candidates = [candidate_record(tokenizer, row['prefix'], ending, context)
                              for ending in [row['positive'], *row['negatives']]]
                if len(candidates) < 4:
                    raise ValueError('Every scene needs three checked negatives')
                self.rows.append(dict(id=row['id'], skill=row['skill'], tokens=tokens,
                                      candidates=candidates))
                if cfg['pair_grouped']:
                    identifier = row.get('pair_id')
                    if not isinstance(identifier, str) or not identifier or 'pair_variant' not in row:
                        raise ValueError('Grouped scenes require pair_id and pair_variant')
                    pair_rows.setdefault(identifier, []).append((len(self.rows)-1, row))
            if not self.rows:
                raise ValueError('Empty scenes')
        self.units = [[i] for i in range(len(self.rows))]
        self.unit_ids = []
        if cfg['pair_grouped']:
            self.units = []
            for identifier, partners in pair_rows.items():
                if len(partners) != 2:
                    raise ValueError(f'Expected exactly two scenes in pair {identifier}')
                (_, first), (_, second) = partners
                if (first['skill'] != second['skill'] or first['pair_variant'] == second['pair_variant']
                        or first['prefix'] == second['prefix'] or first['positive'] == second['positive']
                        or set([first['positive'], *first['negatives']]) !=
                           set([second['positive'], *second['negatives']])):
                    raise ValueError(f'Invalid counterfactual partners in pair {identifier}')
                self.units.append([index for index, _ in partners])
                self.unit_ids.append(identifier)
        self.old_cursor = BatchCursor(math.ceil((len(self.old)-1)/(context-1)), 1, cfg['seed'])
        self.scene_cursor = BatchCursor(max(1, len(self.units)), 1, cfg['seed'] + 100)
        self.generator = np.random.default_rng(cfg['seed'] + 200)
        self.pending = [self.eos]
        self.weights = None
        if cfg['skill_weights']:
            weights = np.asarray([cfg['skill_weights'].get(self.rows[unit[0]]['skill'], 1.)
                                  for unit in self.units])
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError('Skill weights must be finite and strictly positive')
            self.weights = weights / weights.sum()
        if cfg['sampling_manifest']:
            self.weights = self.manifest_weights()
        if state:
            for cursor, name in ((self.old_cursor, 'old'), (self.scene_cursor, 'scene')):
                cursor.epoch, cursor.offset = state[name]['epoch'], state[name]['offset']
            self.generator.bit_generator.state = state['generator']
            self.pending = state['pending']

    def manifest_weights(self):
        """Map frozen training-only mining weights by ID, never by file order."""
        manifest = read_json(self.cfg['sampling_manifest'])
        if manifest.get('schema') != 1:
            raise ValueError('Unknown sampling manifest schema')
        if manifest.get('complete') is not True or manifest.get('pool') != 'train_balanced':
            raise ValueError('Sampling requires completed train_balanced mining')
        for field, filename in (('dataset_sha256', self.cfg['scenes']),
                                ('tokenizer_sha256', self.cfg['tokenizer']),
                                ('checkpoint_sha256', self.cfg['parent'])):
            if manifest.get(field) != sha256(filename):
                raise ValueError(f'Sampling manifest {field} mismatch')
        identifiers, weights = manifest['unit_ids'], np.asarray(manifest['weights'], dtype=np.float64)
        if (len(identifiers) != len(set(identifiers)) or set(identifiers) != set(self.unit_ids)
                or weights.shape != (len(identifiers),)):
            raise ValueError('Sampling manifest must cover every pair unit exactly once')
        if not np.isfinite(weights).all() or (weights <= 0).any() or not np.isclose(weights.sum(), 1., atol=1e-10):
            raise ValueError('Sampling weights must be finite, positive, and sum to one')
        if not set(manifest.get('hard_unit_ids', [])).issubset(set(identifiers)):
            raise ValueError('Hard-unit identifiers are outside the training corpus')
        by_id = dict(zip(identifiers, weights.tolist()))
        ordered = np.asarray([by_id[identifier] for identifier in self.unit_ids])
        skill_mass = {}
        for unit, weight in zip(self.units, ordered):
            skill = self.rows[unit[0]]['skill']
            skill_mass[skill] = skill_mass.get(skill, 0.) + weight
        if len(skill_mass) != 4 or any(not np.isclose(mass, .25, atol=1e-10) for mass in skill_mass.values()):
            raise ValueError('Sampling manifest must assign 25% mass to each of four skills')
        hard = set(manifest['hard_unit_ids'])
        for skill in skill_mass:
            members = [identifier for identifier, unit in zip(self.unit_ids, self.units)
                       if self.rows[unit[0]]['skill'] == skill]
            hard_members = hard.intersection(members)
            for identifier in members:
                expected = (.25/len(members) if not hard_members else
                            .25*(.5/len(members) + (.5/len(hard_members) if identifier in hard else 0.)))
                if not np.isclose(by_id[identifier], expected, rtol=1e-8, atol=1e-12):
                    raise ValueError('Sampling weights differ from fixed 50% uniform / 50% hard recipe')
        # Normalize only roundoff after strict validation, for numpy.choice.
        return ordered / ordered.sum()

    def state(self):
        return dict(old=self.old_cursor.state_dict(), scene=self.scene_cursor.state_dict(),
                    generator=self.generator.bit_generator.state, pending=self.pending)

    def next(self, target_count):
        synthetic = (0 if self.cfg['mode'] in ('control', 'pair_only')
                     else round(target_count*self.cfg['synthetic_fraction']))
        historical = target_count - synthetic
        records, selected = [], []
        left = historical
        while left:
            index = self.old_cursor.next()[0]
            start = index * (self.context - 1)
            # Exact mixture quotas trim the final sampled historical window.
            # This is shuffled-window sampling, not a claim of full epoch coverage.
            tokens = self.old[start:start + min(self.context, left + 1)].astype(np.int64).tolist()
            records.append((tokens, [-100] + tokens[1:]))
            left -= len(tokens) - 1
        left = synthetic
        while left:
            n = min(self.context - 1, left)
            while len(self.pending) < n + 1:
                index = (int(self.generator.choice(len(self.rows), p=self.weights)) if self.weights is not None
                         else self.scene_cursor.next()[0])
                self.pending.extend(self.rows[index]['tokens'])
                selected.append(index)
            tokens = self.pending[:n + 1]
            self.pending = self.pending[n:]
            records.append((tokens, [-100] + tokens[1:]))
            left -= n
        if self.cfg['mode'] == 'pair_only':
            # Same historical LM batches as control. Scenes supply only the
            # contrastive objective; their templated prefixes are never CE targets.
            units_per_step = self.cfg['pair_only_scenes'] // (2 if self.cfg['pair_grouped'] else 1)
            for _ in range(units_per_step):
                index = (int(self.generator.choice(len(self.units), p=self.weights)) if self.weights is not None
                         else self.scene_cursor.next()[0])
                selected.extend(self.units[index])
        batches = [padded(records[i:i+self.cfg['batch_size']], self.pad)
                   for i in range(0, len(records), self.cfg['batch_size'])]
        return batches, selected, synthetic


def backward_pairs(model, stream, selected, device, precision, weight, negatives, batch_scenes, ending_ce_weight=0.):
    if not selected:
        return dict(pair_loss=0., ending_nll=None, ending_ce_targets=0)
    positive_counts = [sum(label != -100 for label in stream.rows[index]['candidates'][0][1][1:])
                       for index in selected]
    ending_targets = sum(positive_counts)
    if ending_targets == 0:
        raise ValueError('Positive endings have no supervised targets')
    loss_sum, ending_sum = 0., 0.
    for i in range(0, len(selected), batch_scenes):
        chunk = selected[i:i+batch_scenes]
        records = [candidate for index in chunk
                   for candidate in stream.rows[index]['candidates'][:negatives+1]]
        ids, labels = (t.to(device) for t in padded(records, stream.pad))
        with autocast(device, precision):
            scores = continuation_scores(model(ids), labels).reshape(len(chunk), negatives+1)
            losses = pair_objective(scores)
        if not torch.isfinite(losses).all():
            raise FloatingPointError('Nonfinite pair objective')
        positive_lengths = scores.new_tensor(positive_counts[i:i+len(chunk)])
        # Existing candidate masks supervise endings only, without final EOS.
        # Sum over positive tokens / ALL step-positive tokens, not mean of means.
        positive_nll_sum = -(scores[:, 0] * positive_lengths).sum()
        objective = weight * losses.sum() / len(selected)
        if ending_ce_weight:
            objective = objective + ending_ce_weight * positive_nll_sum / ending_targets
        objective.backward()
        loss_sum += losses.detach().sum().item()
        ending_sum += positive_nll_sum.detach().item()
    return dict(pair_loss=loss_sum / len(selected), ending_nll=ending_sum / ending_targets,
                ending_ce_targets=ending_targets if ending_ce_weight else 0)


def check_gpu_idle():
    # Shared Windows graphics memory is acceptable; another compute task is not.
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    usage, memory = (int(x.strip()) for x in result.stdout.strip().splitlines()[0].split(','))
    if usage > 15 or memory > 2500:
        raise RuntimeError(f'GPU occupied: utilization={usage}%, memory={memory} MiB; no process stopped')


def train(raw, run_dir, device_name='cuda', resume=None, stop_after=None):
    cfg = config(raw)
    run_dir = Path(run_dir)
    with run_lock(ROOT / 'runs/pretrain_series01', 'series'), run_lock(run_dir, 'training'):
        return train_locked(cfg, run_dir, device_name, resume, stop_after)


def train_locked(cfg, run_dir, device_name, resume, stop_after):
    device, precision = resolve_device(device_name, cfg['precision'])
    if device.type == 'cuda':
        check_gpu_idle()
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError('Refuse to overwrite an existing run')
    if resume and Path(resume).resolve().parent != run_dir.resolve():
        raise ValueError('Resume must belong to selected run')
    torch.set_num_threads(cfg['cpu_threads'])
    seed_everything(cfg['seed'])
    parent = load_checkpoint(cfg['parent'])
    model_cfg = parent['model_config']
    tokenizer_sha = sha256(cfg['tokenizer'])
    if tokenizer_sha != parent['tokenizer_sha256']:
        raise ValueError('Parent tokenizer hash differs')
    tokenizer = Tokenizer.from_file(cfg['tokenizer'])
    if tokenizer.get_vocab_size() != model_cfg['vocab_size']:
        raise ValueError('Tokenizer vocabulary differs from model')
    paths = [cfg['parent'], cfg['historical'], cfg['tokenizer']]
    if cfg['mode'] != 'control':
        paths.append(cfg['scenes'])
    if cfg['sampling_manifest']:
        paths.append(cfg['sampling_manifest'])
    identity = dict(config=cfg, inputs_sha256={str(p): sha256(p) for p in paths},
        code_sha256={**code_hashes(), 'train_causal_series.py': sha256(__file__)},
        environment=environment(device, precision))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    model = GPT(ModelConfig(**model_cfg)).to(device)
    model.load_state_dict(parent['model'])
    optimizer = make_optimizer(model, cfg, device)
    lineage = deepcopy(parent['lineage'])
    lineage['parents'].append(dict(checkpoint_sha256=sha256(cfg['parent']),
        domain='prose', step=parent['step'], optimizer='new AdamW for controlled adaptation'))
    parent_steps = parent.get('parent_steps', 0) + parent['step']
    parent_tokens = parent.get('parent_tokens_seen', 0) + parent['tokens_seen']
    step, tokens_seen, synthetic_seen, consumed_seconds = 0, 0, 0, 0.
    cumulative_pair_scene_count = 0
    ending_ce_tokens_seen = 0
    stream_state = None
    if resume:
        state = load_checkpoint(resume)
        if state.get('fingerprint') != fingerprint:
            raise ValueError('Resume input/config/code/environment fingerprint mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        step, tokens_seen = state['step'], state['tokens_seen']
        synthetic_seen, consumed_seconds = state['synthetic_tokens_seen'], state['consumed_seconds']
        cumulative_pair_scene_count = state.get('cumulative_pair_scene_count', 0)
        ending_ce_tokens_seen = state.get('ending_ce_tokens_seen', 0)
        stream_state, lineage = state['stream'], state['lineage']
        restore_rng(state['rng'])
    del parent
    stream = Stream(cfg, tokenizer, model_cfg['context_length'], stream_state)
    schedule = {**cfg, 'max_steps': math.ceil(cfg['max_tokens']/cfg['target_tokens'])}
    schedule['warmup_steps'] = min(schedule['max_steps']-1, round(schedule['max_steps']*cfg['warmup_fraction']))
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'environment.json', identity)
    metrics = run_dir / 'metrics.jsonl'
    repair_incomplete_jsonl_tail(metrics)
    append_event(metrics, dict(event='resume' if resume else 'start', step=step, fingerprint=fingerprint))
    attempt_start = time.monotonic()
    starting_seconds = consumed_seconds

    def status(name, **extra):
        write_json(run_dir / 'status.json', dict(status=name, step=step, supervised_tokens_seen=tokens_seen,
            synthetic_tokens_seen=synthetic_seen, max_tokens=cfg['max_tokens'],
            cumulative_pair_scene_count=cumulative_pair_scene_count,
            ending_ce_tokens_seen=ending_ce_tokens_seen,
            total_lm_supervised_tokens=tokens_seen+ending_ce_tokens_seen,
            consumed_seconds=starting_seconds+time.monotonic()-attempt_start, **extra))

    def save(name='last.pt'):
        save_checkpoint(run_dir / name, dict(format='poetrygpt-v1', model_config=model_cfg,
            config={**cfg, 'model': model_cfg, 'domain': 'prose'}, model=model.state_dict(),
            optimizer=optimizer.state_dict(), step=step, tokens_seen=tokens_seen,
            synthetic_tokens_seen=synthetic_seen, stream=stream.state(), rng=rng_state(),
            cumulative_pair_scene_count=cumulative_pair_scene_count,
            ending_ce_tokens_seen=ending_ce_tokens_seen,
            lineage=lineage, fingerprint=fingerprint, tokenizer_sha256=tokenizer_sha,
            parent_steps=parent_steps, parent_tokens_seen=parent_tokens,
            consumed_seconds=starting_seconds+time.monotonic()-attempt_start))

    model.train()
    status('running')
    if not resume:
        save()
    try:
        while tokens_seen < cfg['max_tokens']:
            if ((run_dir / 'STOP').exists() or (stop_after is not None and step >= stop_after)
                    or starting_seconds+time.monotonic()-attempt_start >= cfg['max_wall_seconds']):
                save()
                status('stopped_at_update_boundary')
                return run_dir / 'last.pt'
            started = time.monotonic()
            targets = min(cfg['target_tokens'], cfg['max_tokens']-tokens_seen)
            batches, selected, synthetic = stream.next(targets)
            lr = learning_rate(step, schedule)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            # Pairwise dropout cannot change the following update's LM dropout.
            torch.manual_seed(cfg['seed'] + 2*step)
            nll = backward_microbatches(model, batches, targets, device, precision)
            pair_metrics = dict(pair_loss=0., ending_nll=None, ending_ce_targets=0)
            if cfg['mode'] in ('pair', 'pair_only'):
                torch.manual_seed(cfg['seed'] + 2*step + 1)
                pair_metrics = backward_pairs(model, stream, selected, device, precision,
                    cfg['pair_weight'], cfg['negatives'], cfg['pair_batch_scenes'], cfg['ending_ce_weight'])
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            before = tokens_seen
            step += 1
            tokens_seen += targets
            synthetic_seen += synthetic
            if cfg['mode'] in ('pair', 'pair_only'):
                cumulative_pair_scene_count += len(selected)
            ending_ce_tokens_seen += pair_metrics['ending_ce_targets']
            if step % cfg['log_interval'] == 0 or step == 1 or tokens_seen == cfg['max_tokens']:
                event = dict(event='train', step=step, nll=nll, **pair_metrics, lr=lr,
                    grad_norm_before_clip=float(norm), tokens_seen=tokens_seen, synthetic_tokens_seen=synthetic_seen,
                    cumulative_pair_scene_count=cumulative_pair_scene_count,
                    ending_ce_tokens_seen=ending_ce_tokens_seen,
                    seconds=time.monotonic()-started, pair_scenes=len(selected),
                    peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None)
                append_event(metrics, event)
                print(json.dumps(event), flush=True)
            if before < cfg['max_tokens']/2 <= tokens_seen:
                save('half.pt')
            if step % cfg['save_interval'] == 0 or tokens_seen == cfg['max_tokens']:
                save()
            status('running', last_nll=nll)
        status('completed', final_checkpoint='last.pt')
    except BaseException as error:
        # Do not checkpoint possibly half-updated optimizer state on failure.
        status('failed_or_interrupted', error=repr(error), recovery='Resume last.pt; replay unsaved updates')
        raise
    return run_dir / 'last.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    train(read_json(args.config), args.run_dir, args.device, args.resume, args.stop_after)


if __name__ == '__main__':
    main()
