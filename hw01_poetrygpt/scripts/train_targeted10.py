"""Small CE, input-noise or auxiliary offset-two forks of pretrain08.

Default: retain model, optimizer moments, sampler, dropout RNG and absolute token
LR schedule. New local fingerprints never relax the parent's exact-resume rules.
Uniform vocabulary noise is regularization, not meaning-preserving augmentation.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer

from poetrygpt.data import PreparedDataset, read_json, sha256
from poetrygpt.evaluate import evaluate
from poetrygpt.model import GPT, ModelConfig, token_losses
from poetrygpt.runtime import (append_event, autocast, code_hashes, environment, load_checkpoint,
    repair_incomplete_jsonl_tail, resolve_device, restore_rng, rng_state, run_lock,
    save_checkpoint, write_json)
from poetrygpt.train import backward_microbatches
from scripts.train_causal_series import check_gpu_idle, padded
from scripts.train_narrative_series04 import resume_accounting
from scripts.train_pretrain08 import EpochStream, optimizers, token_lr

DEFAULTS = dict(mode='ce', noise_probability=0., noise_seed=2026110410,
    target_tokens=None, batch_size=None, lr_multiplier=1., schedule='inherit',
    lr=None, min_lr=None, warmup_tokens=None, cpu_threads=4,
    max_active_seconds=900, max_cuda_allocated_mib=14000,
    save_interval=100, eval_interval=100, eval_batches=32, eval_batch_size=16,
    log_interval=25, mtp_alpha=0., mtp_ramp_tokens=0,
    adam_beta2=None, weight_decay_multiplier=1., context_overlap=0)
DEPENDENCIES = ('train_pretrain08.py', 'train_narrative_series04.py',
                'train_causal_series.py', 'run_task_pilot02.py', 'training_data.py')


def config(raw):
    if set(raw) - DEFAULTS.keys() - {'parent', 'max_tokens'} or not {'parent', 'max_tokens'} <= raw.keys():
        raise ValueError('Expected parent/max_tokens and supported targeted10 settings')
    cfg = {**deepcopy(DEFAULTS), **deepcopy(raw)}
    if not isinstance(cfg['parent'], str) or not cfg['parent']:
        raise ValueError('Expected parent checkpoint path')
    for key in ('max_tokens', 'cpu_threads', 'max_active_seconds', 'max_cuda_allocated_mib',
                'save_interval', 'eval_interval', 'eval_batches', 'eval_batch_size', 'log_interval'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError('Expected positive integer: ' + key)
    for key in ('target_tokens', 'batch_size'):
        if cfg[key] is not None and (type(cfg[key]) is not int or cfg[key] < 1):
            raise ValueError('Expected positive integer or null: ' + key)
    if type(cfg['context_overlap']) is not int or cfg['context_overlap'] < 0:
        raise ValueError('context_overlap must be a nonnegative integer')
    if type(cfg['noise_seed']) is not int or not 0 <= cfg['noise_seed'] < 2**32:
        raise ValueError('Invalid noise seed')
    for key in ('noise_probability', 'lr_multiplier', 'mtp_alpha', 'weight_decay_multiplier'):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]):
            raise ValueError('Expected finite scalar: ' + key)
    if cfg['adam_beta2'] is not None and (type(cfg['adam_beta2']) not in (int, float)
            or not math.isfinite(cfg['adam_beta2']) or not 0 < cfg['adam_beta2'] < 1):
        raise ValueError('adam_beta2 must be null or strictly between zero and one')
    if cfg['weight_decay_multiplier'] <= 0:
        raise ValueError('weight_decay_multiplier must be positive')
    if cfg['mode'] not in ('ce', 'noise', 'mtp2') or not 0 <= cfg['noise_probability'] <= .2:
        raise ValueError('Use ce/noise/mtp2 and at most 20% input replacement')
    if cfg['mode'] != 'noise' and cfg['noise_probability'] != 0:
        raise ValueError('Only noise mode may corrupt inputs')
    if (not 0 <= cfg['mtp_alpha'] <= .5 or type(cfg['mtp_ramp_tokens']) is not int
            or not 0 <= cfg['mtp_ramp_tokens'] <= cfg['max_tokens']):
        raise ValueError('MTP requires alpha in [0,.5] and bounded integer ramp')
    if cfg['mode'] != 'mtp2' and (cfg['mtp_alpha'] != 0 or cfg['mtp_ramp_tokens'] != 0):
        raise ValueError('Only mtp2 mode may enable auxiliary prediction')
    if cfg['lr_multiplier'] <= 0 or cfg['schedule'] not in ('inherit', 'restart'):
        raise ValueError('Invalid LR multiplier/schedule')
    if cfg['schedule'] == 'inherit':
        if any(cfg[k] is not None for k in ('lr', 'min_lr', 'warmup_tokens')):
            raise ValueError('Inherited schedule cannot silently override LR/warmup; use multiplier or explicit restart')
    else:
        if (any(type(cfg[k]) not in (int, float) or not math.isfinite(cfg[k]) for k in ('lr', 'min_lr'))
                or not 0 < cfg['min_lr'] <= cfg['lr']
                or type(cfg['warmup_tokens']) is not int
                or not 0 <= cfg['warmup_tokens'] < cfg['max_tokens']):
            raise ValueError('Restart requires explicit lr/min_lr/warmup_tokens')
    return cfg


class OverlapEpochStream(EpochStream):
    """Optional extra same-work context; original windows and target order survive.

    Reserve overlap positions by splitting a window into smaller target pieces.
    Context-only tokens carry no main loss. The inherited state still identifies
    the exact next target, so a parent stream can be forked without reordering it.
    """
    def __init__(self, data, seed, state=None, overlap=0):
        if type(overlap) is not int or not 0 <= overlap < data.context_length - 1:
            raise ValueError('context_overlap must be smaller than context_length - 1')
        self.overlap = overlap
        super().__init__(data, seed, state)

    def next(self, quota, batch_size):
        if self.overlap == 0:
            return super().next(quota, batch_size)
        if type(quota) is not int or quota < 1 or batch_size < 1:
            raise ValueError('Positive quota and batch size required')
        records = []
        context = self.data.context_length
        eos = self.data.manifest['special_tokens']['<|eos|>']
        while quota:
            if self.pending is None:
                if self.cursor.offset == self.cursor.size:
                    self._set_epoch(self.cursor.epoch + 1)
                self.pending = dict(index=self.cursor.next()[0], consumed=0)
            start, end, owner = map(int, self.windows[self.pending['index']])
            first = start + 1 + self.pending['consumed']
            stop = min(end, first + quota, first + context - 1 - self.overlap)
            input_begin = max(self.data.metadata[owner]['start'] - 1, stop - context)
            if not input_begin <= first - 1 < stop - 1:
                raise ValueError('Invalid overlap window or pending target')
            ids = list(map(int, self.data.ids[input_begin:stop]))
            if eos in ids[1:-1]:
                raise ValueError('Overlap context crosses an interior EOS')
            labels = [-100] * (first - input_begin) + ids[first - input_begin:]
            records.append((ids, labels))
            count = stop - first
            self.total_targets += count
            self.record_targets[owner] += count
            self.pending['consumed'] += count
            quota -= count
            if stop == end:
                self.pending = None
        return [padded(records[i:i + batch_size], self.data.pad_id)
                for i in range(0, len(records), batch_size)]


class InputNoise:
    """CPU generator isolated from dropout RNG; labels are never modified."""
    def __init__(self, tokenizer, special_ids, probability, seed, state=None):
        vocabulary = tokenizer.get_vocab_size()
        added = {i for i, value in tokenizer.get_added_tokens_decoder().items() if value.special}
        protected = set(special_ids) | added
        protected.update(i for i in range(vocabulary)
                         if any(c in tokenizer.decode([i], skip_special_tokens=False) for c in '\r\n'))
        self.allowed = torch.ones(vocabulary, dtype=torch.bool)
        self.allowed[list(protected)] = False
        self.pool = torch.where(self.allowed)[0]
        if not len(self.pool):
            raise ValueError('No non-special/non-newline replacement tokens')
        self.probability = probability
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.counts = dict(input_positions=0, eligible=0, selected=0, changed=0)
        if state is not None:
            self.generator.set_state(state['generator'])
            self.counts = deepcopy(state['counts'])

    def state(self):
        return dict(generator=self.generator.get_state(), counts=deepcopy(self.counts))

    def apply(self, batches):
        if self.probability == 0:
            return batches
        corrupted = []
        for ids, labels in batches:
            if ids.device.type != 'cpu':
                raise ValueError('Corrupt CPU input batches before transferring to the device')
            eligible = self.allowed[ids]
            selected = eligible & (torch.rand(ids.shape, generator=self.generator) < self.probability)
            count = int(selected.sum())
            changed = ids.clone()
            if count:
                replacement = self.pool[torch.randint(len(self.pool), (count,), generator=self.generator)]
                changed[selected] = replacement
            self.counts['input_positions'] += ids.numel()
            self.counts['eligible'] += int(eligible.sum())
            self.counts['selected'] += count
            self.counts['changed'] += int((changed != ids).sum())
            corrupted.append((changed, labels))
        return corrupted


def future_labels(ids, labels, boundary_ids):
    """h[t] predicts x[t+2]; exclude an intervening document/padding boundary.

    The initial BOS/EOS can condition the second word of its new document.
    A final EOS target is valid; predicting *across* that EOS is not.
    """
    targets = labels[:, 2:].clone()
    for boundary in boundary_ids:
        targets[ids[:, 1:-1] == boundary] = -100
    return targets


class MTP2:
    """Training-only output head; the base model/state_dict remain unchanged.

    Inspired by independent future heads (arxiv.org/abs/2404.19737), not a
    replication of BabyLM2025 curricula (aclanthology.org/2025.babylm-main.41/).
    """
    def __init__(self, model, recipe, tokenizer_sha, boundary_ids, state=None):
        device = model.tokens.weight.device
        # Linear initializes on CPU; isolate that RNG before copying tied weights.
        with torch.random.fork_rng(devices=[]):
            self.head = nn.Linear(model.config.d_model, model.config.vocab_size, bias=False)
        self.head = self.head.to(device)
        with torch.no_grad():
            self.head.weight.copy_(model.tokens.weight)
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=recipe['lr'],
            betas=tuple(recipe['betas']), eps=recipe['eps'], weight_decay=recipe['weight_decay'],
            fused=bool(recipe['fused_optimizer'] and device.type == 'cuda'))
        self.boundary_ids = tuple(sorted(set(boundary_ids)))
        self.tokenizer_sha = tokenizer_sha
        self.targets_seen = 0
        if state is not None:
            if (state['tokenizer_sha256'] != tokenizer_sha
                    or state['head_shape'] != list(self.head.weight.shape)
                    or state['boundary_ids'] != list(self.boundary_ids)):
                raise ValueError('MTP head/tokenizer/boundary mismatch')
            self.head.load_state_dict(state['head'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.targets_seen = state['targets_seen']

    def state(self):
        return dict(head=self.head.state_dict(), optimizer=self.optimizer.state_dict(),
            head_shape=list(self.head.weight.shape), tokenizer_sha256=self.tokenizer_sha,
            boundary_ids=list(self.boundary_ids), targets_seen=self.targets_seen,
            method='independent_linear_offset2_output_head_initialized_from_tied_embedding')

    def backward(self, model, batches, denominator, device, precision, alpha):
        count = (sum(int(future_labels(ids, labels, self.boundary_ids).ne(-100).sum())
                     for ids, labels in batches) if alpha else 0)
        if not count:
            nll = backward_microbatches(model, batches, denominator, device, precision)
            return dict(main_nll=nll, auxiliary_nll=None, auxiliary_targets=0,
                        alpha=alpha, total_loss=nll)
        main_sum = auxiliary_sum = 0.
        for ids, labels in batches:
            targets = future_labels(ids, labels, self.boundary_ids).to(device)
            valid = targets.ne(-100)
            ids, labels = ids.to(device), labels.to(device)
            hidden = []
            handle = model.final_norm.register_forward_hook(lambda module, inputs, output: hidden.append(output))
            try:
                with autocast(device, precision):
                    logits = model(ids)
                    losses, _ = token_losses(logits, labels)
                    main = losses.sum()
                    if len(hidden) != 1:
                        raise RuntimeError('Expected one final_norm output per base forward')
                    auxiliary = (F.cross_entropy(self.head(hidden[0][:, :-2][valid]).float(),
                                 targets[valid], reduction='sum') if valid.any() else main.new_zeros(()))
                    objective = main / denominator + alpha * auxiliary / count
            finally:
                handle.remove()
            if not torch.isfinite(objective):
                raise FloatingPointError('Nonfinite MTP objective')
            objective.backward()
            main_sum += main.detach().item()
            auxiliary_sum += auxiliary.detach().item()
        self.targets_seen += count
        main_nll, auxiliary_nll = main_sum / denominator, auxiliary_sum / count
        return dict(main_nll=main_nll, auxiliary_nll=auxiliary_nll,
                    auxiliary_targets=count, alpha=alpha, total_loss=main_nll + alpha * auxiliary_nll)


def mtp_weight(local_end, cfg):
    ramp = cfg['mtp_ramp_tokens']
    return cfg['mtp_alpha'] * min(1., local_end / ramp) if ramp else cfg['mtp_alpha']


def pilot_lr(local_end, parent_tokens, parent_recipe, cfg):
    if cfg['schedule'] == 'inherit':
        value = token_lr(parent_tokens + local_end, parent_recipe)
    else:
        schedule = dict(lr=cfg['lr'], min_lr=cfg['min_lr'], warmup_tokens=cfg['warmup_tokens'],
                        schedule_tokens=cfg['max_tokens'])
        value = token_lr(local_end, schedule)
    return cfg['lr_multiplier'] * value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def optimizer_group_settings(states):
    """Record immutable parent hyperparameters, without copying moments/weights."""
    return {name: [dict(weight_decay=group['weight_decay'],
                       **({'betas': list(group['betas'])} if 'betas' in group else {}))
                   for group in state['param_groups']] for name, state in states.items()}


def apply_optimizer_overrides(opts, parent_groups, cfg):
    """Apply once from immutable parent values, also after loading pilot state.

    Small-batch token-half-life motivation: arxiv.org/html/2507.07101v1 sec4.3.
    Late continuation with inherited moments/mixed Muon is our hypothesis, not
    that paper's experiment. Beta1 and Muon momentum are intentionally unchanged.
    """
    if opts.keys() != parent_groups.keys():
        raise ValueError('Optimizer group identities differ from parent')
    for name, opt in opts.items():
        source = parent_groups[name]
        if len(source) != len(opt.param_groups):
            raise ValueError('Optimizer group count differs from parent')
        for group, original in zip(opt.param_groups, source):
            group['weight_decay'] = original['weight_decay'] * cfg['weight_decay_multiplier']
            if isinstance(opt, torch.optim.AdamW):
                beta1, beta2 = original['betas']
                group['betas'] = (beta1, beta2 if cfg['adam_beta2'] is None else cfg['adam_beta2'])


def train(raw, run_dir, device_name='cuda', resume=False, stop_after=None):
    cfg, run_dir = config(raw), Path(run_dir)
    if stop_after is not None and (type(stop_after) is not int or stop_after < 0):
        raise ValueError('stop_after must be a nonnegative complete local update')
    with run_lock(ROOT / 'runs/pretrain_series01', 'series'), run_lock(run_dir, 'training'):
        return train_locked(cfg, run_dir, device_name, resume, stop_after)


def train_locked(cfg, run_dir, device_name, resume, stop_after):
    started = time.monotonic()
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError('Refuse to overwrite a targeted run')
    state = load_checkpoint(run_dir / 'last.pt' if resume else cfg['parent'])
    if resume:
        if state.get('pilot_config') != cfg:
            raise ValueError('Exact pilot resume requires unchanged configuration')
        parent_recipe, parent_record = state['parent_recipe'], state['parent_record']
    else:
        if (state['config'].get('domain') != 'prose' or not state.get('optimizers')
                or state.get('pilot_config') is not None):
            raise ValueError('Fork requires a full pretrain08 prose checkpoint')
        parent_recipe = deepcopy(state['config'])
        parent_path = Path(cfg['parent'])
        parent_directory = parent_path.parent
        # Frozen milestones live in run/stages/*.pt; their identity belongs to
        # run/environment.json. No arbitrary ancestor search/fallback is allowed.
        if parent_directory.name == 'stages':
            parent_directory = parent_directory.parent
        environment_path = parent_directory / 'environment.json'
        parent_environment = read_json(environment_path)
        if (digest(parent_environment) != state['fingerprint']
                or {**parent_environment['config'], 'domain': 'prose'} != parent_recipe):
            raise ValueError('Parent environment/checkpoint fingerprint mismatch')
        parent_record = dict(path=cfg['parent'], sha256=sha256(cfg['parent']),
            fingerprint=state['fingerprint'], tokens=state['tokens_seen'], step=state['step'],
            environment_path=str(environment_path), environment_sha256=sha256(environment_path),
            optimizer_groups=optimizer_group_settings(state['optimizers']))
    recipe = deepcopy(parent_recipe)
    recipe['weight_decay'] = parent_recipe['weight_decay'] * cfg['weight_decay_multiplier']
    if cfg['adam_beta2'] is not None:
        recipe['betas'] = [parent_recipe['betas'][0], cfg['adam_beta2']]
    for key in ('target_tokens', 'batch_size'):
        if cfg[key] is not None:
            recipe[key] = cfg[key]
    torch.set_num_threads(cfg['cpu_threads'])
    device, precision = resolve_device(device_name, recipe['precision'])
    if device.type == 'cuda':
        check_gpu_idle()
        available = torch.cuda.get_device_properties(device).total_memory
        fraction = min(1., cfg['max_cuda_allocated_mib'] * 1024**2 / available)
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(fraction, device_index)
        torch.cuda.reset_peak_memory_stats(device)
    training = PreparedDataset(recipe['data'], 'prose', 'train', recipe['model']['context_length'])
    validation = PreparedDataset(recipe['data'], 'prose', 'validation', recipe['model']['context_length'])
    tokenizer_sha = sha256(recipe['tokenizer'])
    manifest_sha = sha256(Path(recipe['data']) / 'manifest.json')
    if tokenizer_sha != state['tokenizer_sha256'] or manifest_sha != state['prepared_manifest_sha256']:
        raise ValueError('Parent data/tokenizer differs from pilot inputs')
    tokenizer = Tokenizer.from_file(recipe['tokenizer'])
    if tokenizer.get_vocab_size() != recipe['model']['vocab_size'] or recipe['model'] != state['model_config']:
        raise ValueError('Model/tokenizer configuration mismatch')
    identity = dict(pilot_config=cfg, parent=parent_record, recipe=recipe,
        tokenizer_sha256=tokenizer_sha, prepared_manifest_sha256=manifest_sha,
        inputs_sha256={**training.hashes, **validation.hashes}, code_sha256=code_hashes(),
        trainer_sha256=sha256(__file__),
        helpers_sha256={name: sha256(ROOT / 'scripts' / name) for name in DEPENDENCIES},
        environment=environment(device, precision))
    fingerprint = digest(identity)
    if resume and state['fingerprint'] != fingerprint:
        raise ValueError('Exact pilot resume requires unchanged code/data/environment')
    model = GPT(ModelConfig(**recipe['model'])).to(device)
    model.load_state_dict(state['model'])
    opts = optimizers(model, recipe, device)
    if set(opts) != set(state['optimizers']):
        raise ValueError('Pilot must preserve optimizer family and every optimizer state')
    for name, opt in opts.items():
        opt.load_state_dict(state['optimizers'][name])
    apply_optimizer_overrides(opts, parent_record['optimizer_groups'], cfg)
    mtp = None
    if cfg['mode'] == 'mtp2':
        if resume and 'mtp2' not in state:
            raise ValueError('MTP resume requires the auxiliary head and optimizer')
        boundary_ids = [training.manifest['special_tokens'][key] for key in ('<|bos|>', '<|eos|>', '<|pad|>')]
        mtp = MTP2(model, recipe, tokenizer_sha, boundary_ids, state.get('mtp2') if resume else None)
        apply_optimizer_overrides({'mtp2': mtp.optimizer}, {'mtp2': [dict(
            weight_decay=parent_recipe['weight_decay'], betas=parent_recipe['betas'])]}, cfg)
    elif resume and 'mtp2' in state:
        raise ValueError('CE/noise resume must not discard an auxiliary head')
    trainable = list(model.parameters()) + (list(mtp.head.parameters()) if mtp else [])
    stream = OverlapEpochStream(training, recipe['seed'], state['stream'], cfg['context_overlap'])
    local_tokens, step = (state['local_tokens_seen'], state['step']) if resume else (0, 0)
    if (stream.total_targets != parent_record['tokens'] + local_tokens
            or stream.total_targets != state['tokens_seen'] or local_tokens > cfg['max_tokens']):
        raise ValueError('Parent/pilot/stream counters disagree')
    noise = InputNoise(tokenizer, training.manifest['special_tokens'].values(),
        cfg['noise_probability'], cfg['noise_seed'], state.get('noise') if resume else None)
    lineage = deepcopy(state['lineage'])
    if not resume:
        lineage['parents'].append(dict(checkpoint_sha256=parent_record['sha256'], domain='prose',
            step=parent_record['step'], optimizer='inherited complete moments and stream',
            pilot=cfg['mode'], schedule=cfg['schedule'], lr_multiplier=cfg['lr_multiplier']))
    best_nll = state.get('best_nll') if resume else None
    previous_seconds, attempt = resume_accounting(run_dir, fingerprint, state) if resume else (0., 1)
    restore_rng(state['rng'])
    del state
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'config.json', cfg)
    write_json(run_dir / 'environment.json', identity)
    metrics = run_dir / 'metrics.jsonl'
    repair_incomplete_jsonl_tail(metrics)

    def elapsed():
        return previous_seconds + time.monotonic() - started

    def event(name, **extra):
        append_event(metrics, dict(event=name, step=step, fingerprint=fingerprint,
            attempt_id=attempt, consumed_seconds=elapsed(), local_targets=local_tokens,
            absolute_targets=parent_record['tokens'] + local_tokens, **extra))

    def status(name, **extra):
        write_json(run_dir / 'status.json', dict(status=name, step=step,
            fingerprint=fingerprint, attempt_id=attempt, consumed_seconds=elapsed(),
            supervised_tokens_seen=stream.total_targets, local_tokens_seen=local_tokens,
            absolute_targets=stream.total_targets, local_target_budget=cfg['max_tokens'],
            noise=noise.counts, auxiliary_targets_seen=mtp.targets_seen if mtp else 0, **extra))

    def save(name='last.pt'):
        if any(not torch.isfinite(p).all() for p in trainable):
            raise FloatingPointError('Refuse nonfinite pilot weights')
        save_checkpoint(run_dir / name, dict(format='poetrygpt-v1', model=model.state_dict(),
            model_config=recipe['model'], config=recipe, pilot_config=cfg,
            parent_recipe=parent_recipe, parent_record=parent_record,
            optimizers={k: opt.state_dict() for k, opt in opts.items()},
            tokens_seen=stream.total_targets, absolute_tokens_seen=stream.total_targets,
            local_tokens_seen=local_tokens, step=step, parent_steps=parent_record['step'],
            stream=stream.state(), rng=rng_state(), noise=noise.state(), lineage=lineage,
            fingerprint=fingerprint, tokenizer_sha256=tokenizer_sha,
            prepared_manifest_sha256=manifest_sha, best_nll=best_nll,
            consumed_seconds=elapsed(), attempt_id=attempt,
            **({'mtp2': mtp.state()} if mtp else {})))

    event('resume' if resume else 'fork', inherited_optimizer=True,
        rollback_after_step=step if resume else None,
        lr_schedule=cfg['schedule'], lr_multiplier=cfg['lr_multiplier'],
        target_tokens=recipe['target_tokens'], batch_size=recipe['batch_size'],
        adam_beta2=cfg['adam_beta2'], weight_decay_multiplier=cfg['weight_decay_multiplier'],
        context_overlap=cfg['context_overlap'],
        optimizer_groups=optimizer_group_settings({name: opt.state_dict() for name, opt in opts.items()}),
        auxiliary_optimizer_groups=optimizer_group_settings({'mtp2': mtp.optimizer.state_dict()}) if mtp else None)
    model.train()
    if not resume:
        save()
    status('running')
    try:
        while local_tokens < cfg['max_tokens']:
            if ((run_dir / 'STOP').exists() or elapsed() >= cfg['max_active_seconds']
                    or (stop_after is not None and step >= stop_after)):
                save()
                status('stopped_at_update_boundary')
                return run_dir / 'last.pt'
            count = min(recipe['target_tokens'], cfg['max_tokens'] - local_tokens)
            batches = stream.next(count, recipe['batch_size'])
            if sum(int(y[:, 1:].ne(-100).sum()) for _, y in batches) != count:
                raise AssertionError('Supervision quota mismatch')
            corrupted = noise.apply(batches)
            lr = pilot_lr(local_tokens + count, parent_record['tokens'], parent_recipe, cfg)
            for opt in opts.values():
                for group in opt.param_groups:
                    group['lr'] = lr
                opt.zero_grad(set_to_none=True)
            if mtp:
                for group in mtp.optimizer.param_groups:
                    group['lr'] = lr
                mtp.optimizer.zero_grad(set_to_none=True)
                objective = mtp.backward(model, corrupted, count, device, precision,
                                        mtp_weight(local_tokens + count, cfg))
            else:
                main = backward_microbatches(model, corrupted, count, device, precision)
                objective = dict(main_nll=main, auxiliary_nll=None, auxiliary_targets=0,
                                 alpha=0., total_loss=main)
            nll = objective['main_nll']
            norm = torch.nn.utils.clip_grad_norm_(trainable, recipe['grad_clip'], error_if_nonfinite=True)
            for opt in opts.values():
                opt.step()
            if mtp and objective['auxiliary_targets']:
                mtp.optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            step += 1
            local_tokens += count
            peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0.
            event('train', nll=nll, lr=lr, supervised_targets=count,
                grad_norm_before_clip=float(norm), peak_cuda_allocated_mib=peak,
                noise=deepcopy(noise.counts), **objective)
            if peak > cfg['max_cuda_allocated_mib']:
                save()
                status('memory_limit_reached', peak_cuda_allocated_mib=peak)
                return run_dir / 'last.pt'
            if step % cfg['eval_interval'] == 0 or local_tokens == cfg['max_tokens']:
                report = evaluate(model, validation, device, precision, cfg['eval_batch_size'], cfg['eval_batches'])
                score = report['metrics']['overall']['nll']
                write_json(run_dir / 'validation_history' / f'step_{step:08d}.json', dict(step=step, **report))
                event('validation', **report['metrics']['overall'])
                if best_nll is None or score < best_nll:
                    best_nll = score
                    save('best.pt')
            if step % cfg['save_interval'] == 0 or local_tokens == cfg['max_tokens']:
                save()
            if step % cfg['log_interval'] == 0 or local_tokens == cfg['max_tokens']:
                status('running', last_train_nll=nll)
                print(f'step={step} targets={local_tokens} nll={nll:.4f} seconds={elapsed():.1f}', flush=True)
        status('completed')
    except BaseException as error:
        event('failure', error=repr(error))
        status('failed_or_interrupted', error=repr(error), recovery='Resume last.pt; replay unsaved updates')
        raise
    return run_dir / 'last.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    train(read_json(args.config), args.run_dir, args.device, args.resume)


if __name__ == '__main__':
    main()
