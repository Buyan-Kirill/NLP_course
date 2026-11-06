"""Token-budgeted poetry adaptation from a selected pretrain08 checkpoint.

Modes are pure poetry or 90/10 poetry/prose by supervised targets. Both use
one fixed token LR schedule; --until-tokens stops at a comparison milestone.
All declared milestones split optimizer updates even in an uninterrupted run.
Starting a new stage imports weights/ancestry and creates fresh optimizers. Resume
restores the complete local stage without reopening its parent checkpoint.
"""
import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tokenizers import Tokenizer

from poetrygpt.data import PreparedDataset, read_json, sha256
from poetrygpt.evaluate import evaluate
from poetrygpt.model import GPT, ModelConfig
from poetrygpt.runtime import (append_event, code_hashes, environment, load_checkpoint,
    repair_incomplete_jsonl_tail, resolve_device, restore_rng, run_lock,
    save_checkpoint, write_json)
from poetrygpt.train import backward_microbatches
from scripts.train_causal_series import check_gpu_idle
from scripts.train_narrative_series04 import resume_accounting
from scripts.train_pretrain08 import EpochStream, optimizers, token_lr


REQUIRED = {'parent', 'poetry_data', 'prose_data', 'tokenizer', 'model', 'schedule_tokens', 'input_freeze'}
DEFAULTS = dict(domain='poetry', mode='pure', optimizer='adamw', seed=2026110308,
    target_tokens=16384, batch_size=32, warmup_tokens=10000, milestones=None,
    lr=3e-5, min_lr=3e-6, weight_decay=.1, betas=[.9, .95], eps=1e-8,
    grad_clip=1., precision='bf16', cpu_threads=4, fused_optimizer=True,
    save_interval=100, eval_interval=100, eval_batches=0, eval_batch_size=16,
    log_interval=25, max_active_seconds=21600)
DEPENDENCIES = ('scripts/train_pretrain08.py', 'scripts/train_narrative_series04.py',
    'scripts/train_causal_series.py', 'scripts/run_task_pilot02.py', 'scripts/training_data.py')


def config(raw):
    missing = REQUIRED - raw.keys()
    unknown = raw.keys() - REQUIRED - DEFAULTS.keys()
    if missing or unknown:
        raise ValueError(f'Invalid poetry08 config: missing={sorted(missing)}, unknown={sorted(unknown)}')
    cfg = {**deepcopy(DEFAULTS), **deepcopy(raw)}
    cfg['model'] = ModelConfig(**cfg['model']).__dict__.copy()
    if cfg['domain'] != 'poetry' or cfg['mode'] not in ('pure', '90_10'):
        raise ValueError('Expected domain poetry and mode pure or 90_10')
    if cfg['optimizer'] not in ('adamw', 'muon'):
        raise ValueError('Expected optimizer adamw or muon')
    for key in ('parent', 'poetry_data', 'prose_data', 'tokenizer', 'input_freeze'):
        if not isinstance(cfg[key], str) or not cfg[key]:
            raise ValueError('Expected path string: ' + key)
    for key in ('target_tokens', 'batch_size', 'schedule_tokens', 'cpu_threads',
                'save_interval', 'eval_interval', 'eval_batch_size', 'log_interval', 'max_active_seconds'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError('Expected positive integer: ' + key)
    for key in ('warmup_tokens', 'eval_batches', 'seed'):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError('Expected nonnegative integer: ' + key)
    if cfg['warmup_tokens'] >= cfg['schedule_tokens'] or cfg['seed'] >= 2**32 - 1:
        raise ValueError('Invalid warmup or seed (prose uses seed + 1)')
    milestones = cfg['milestones']
    if milestones is None:
        milestones = [cfg['schedule_tokens']]
    if (not isinstance(milestones, (list, tuple)) or not milestones
            or any(type(t) is not int or not 0 < t <= cfg['schedule_tokens'] for t in milestones)):
        raise ValueError('Milestones must be positive integers within the fixed token schedule')
    cfg['milestones'] = sorted(set(milestones))
    if cfg['milestones'][-1] != cfg['schedule_tokens']:
        raise ValueError('Milestones must end at schedule_tokens')
    for key in ('lr', 'min_lr', 'weight_decay', 'eps', 'grad_clip'):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]):
            raise ValueError('Expected finite scalar: ' + key)
    if not 0 < cfg['min_lr'] <= cfg['lr'] or cfg['weight_decay'] < 0 or min(cfg['eps'], cfg['grad_clip']) <= 0:
        raise ValueError('Invalid AdamW/schedule values')
    if not isinstance(cfg['betas'], (list, tuple)) or len(cfg['betas']) != 2 or any(
            type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x < 1 for x in cfg['betas']):
        raise ValueError('Invalid AdamW betas')
    cfg['betas'] = list(cfg['betas'])
    if cfg['precision'] not in ('fp32', 'bf16') or type(cfg['fused_optimizer']) is not bool:
        raise ValueError('Expected fp32/bf16 and boolean fused_optimizer')
    return cfg


def update_quota(tokens_seen, cfg):
    """Fixed update partition, independent of the requested stopping milestone."""
    if type(tokens_seen) is not int or not 0 <= tokens_seen < cfg['schedule_tokens']:
        raise ValueError('Update starts outside the fixed token schedule')
    boundary = next(t for t in cfg['milestones'] if t > tokens_seen)
    return min(cfg['target_tokens'], boundary - tokens_seen)


def validate_checkpoint(state, cfg, tokenizer_sha, *, initial=False):
    """Shared model/tokenizer contract; optimizer formats differ across stages."""
    if (state.get('format') != 'poetrygpt-v1'
            or state.get('lineage', {}).get('origin') != 'random_initialization'):
        raise ValueError('Expected project random-origin ancestry')
    if ModelConfig(**state['model_config']).__dict__ != cfg['model']:
        raise ValueError('Checkpoint model configuration/architecture mismatch')
    if state['tokenizer_sha256'] != tokenizer_sha:
        raise ValueError('Checkpoint tokenizer mismatch')
    if initial and (state['config'].get('domain') != 'prose' or not state.get('optimizers')):
        raise ValueError('Initial parent must be a pretrain08 prose checkpoint with optimizers')
    if not initial and state['config'].get('domain') != 'poetry':
        raise ValueError('Resume requires a local poetry checkpoint')


def load_datasets(cfg, tokenizer_sha):
    """Pure still evaluates prose retention, but never opens prose train arrays."""
    tokenizer = Tokenizer.from_file(cfg['tokenizer'])
    if tokenizer.get_vocab_size() != cfg['model']['vocab_size']:
        raise ValueError('Model/tokenizer vocabulary mismatch')
    training, validation = {}, {}
    for domain in ('poetry', 'prose'):
        folder = cfg[domain + '_data']
        validation[domain] = PreparedDataset(folder, domain, 'validation', cfg['model']['context_length'])
        if domain == 'poetry' or cfg['mode'] == '90_10':
            training[domain] = PreparedDataset(folder, domain, 'train', cfg['model']['context_length'])
    for data in [*training.values(), *validation.values()]:
        declared = data.manifest['inputs_sha256']
        hashes = [v for k, v in declared.items() if k.replace('\\', '/').endswith('tokenizer.json')]
        if hashes != [tokenizer_sha]:
            raise ValueError('Prepared data tokenizer mismatch')
        if any(tokenizer.token_to_id(name) != value for name, value in data.manifest['special_tokens'].items()):
            raise ValueError('Prepared data special token IDs mismatch')
    validate_input_freeze(cfg, tokenizer_sha, training, validation)
    return training, validation


def optimizer_checkpoint(opts, optimizer_name):
    """Retain the single-AdamW checkpoint layout; Muon also needs AdamW state."""
    if optimizer_name == 'adamw':
        return {'optimizer': opts['adamw'].state_dict()}
    return {'optimizers': {name: opt.state_dict() for name, opt in opts.items()}}


def restore_optimizers(opts, state, optimizer_name):
    if state['config'].get('optimizer', 'adamw') != optimizer_name:
        raise ValueError('Resume optimizer changed')
    if optimizer_name == 'adamw':
        if 'optimizer' not in state or 'optimizers' in state:
            raise ValueError('Expected single AdamW optimizer state')
        saved = {'adamw': state['optimizer']}
    else:
        saved = state.get('optimizers', {})
        if 'optimizer' in state or set(saved) != set(opts):
            raise ValueError('Resume requires both Muon and AdamW optimizer states')
    for name, opt in opts.items():
        opt.load_state_dict(saved[name])


def validate_input_freeze(cfg, tokenizer_sha, training, validation):
    """Bind even the first run to the independently reviewed input freeze.

    Frozen relative dataset paths are project-relative. Pure checks prose train
    declarations in the fixed manifest, but only hashes files actually loaded
    for this run (poetry train and both validation sets).
    """
    payload = Path(cfg['input_freeze']).read_bytes()
    freeze = json.loads(payload.decode('utf-8-sig'))
    if freeze.get('status') != 'inputs_verified_not_authorization_to_train':
        raise ValueError('Input freeze has not verified the data')
    if freeze.get('tokenizer_sha256') != tokenizer_sha:
        raise ValueError('Input freeze tokenizer mismatch')
    if set(freeze.get('datasets', {})) != {'poetry', 'prose'}:
        raise ValueError('Input freeze requires poetry and prose datasets')
    expected_training = {'poetry', 'prose'} if cfg['mode'] == '90_10' else {'poetry'}
    if set(training) != expected_training or set(validation) != {'poetry', 'prose'}:
        raise ValueError('Loaded domains differ from the frozen comparison')
    for domain in ('poetry', 'prose'):
        frozen = freeze['datasets'][domain]
        folder = Path(cfg[domain + '_data']).resolve()
        if folder != (ROOT / frozen['path']).resolve():
            raise ValueError('Input freeze dataset path mismatch: ' + domain)
        if sha256(folder / 'manifest.json') != frozen['manifest_sha256']:
            raise ValueError('Input freeze manifest mismatch: ' + domain)
        manifest = validation[domain].manifest
        if (frozen['context_length'] != cfg['model']['context_length']
                or manifest['context_length'] != frozen['context_length']):
            raise ValueError('Input freeze context mismatch: ' + domain)
        if (manifest['special_tokens'] != frozen['special_tokens']
                or frozen['special_tokens'] != freeze['datasets']['poetry']['special_tokens']):
            raise ValueError('Input freeze special token IDs mismatch: ' + domain)
        for split in ('train', 'validation'):
            expected = frozen['splits'][split]['files_sha256']
            names = {f'{domain}_{split}.npy', f'{domain}_{split}_offsets.json'}
            if set(expected) != names or any(manifest['output_sha256'].get(name) != digest
                                              for name, digest in expected.items()):
                raise ValueError(f'Input freeze file declarations mismatch: {domain}/{split}')
            data = validation[domain] if split == 'validation' else training.get(domain)
            if data is not None and (data.folder.resolve() != folder or data.hashes != expected):
                raise ValueError(f'Loaded files differ from input freeze: {domain}/{split}')
    return hashlib.sha256(payload).hexdigest()


class MixtureStream:
    """Cumulative integer quotas; every partially consumed domain window survives.

    At total N targets, prose has floor(N * numerator / denominator). Computing
    a difference of cumulative counts avoids rounding away small-batch tails.
    Each domain delegates its window/epoch and pending-tail state to EpochStream.
    """
    def __init__(self, datasets, seed, prose_numerator, prose_denominator=10, state=None):
        if (type(prose_numerator) is not int or type(prose_denominator) is not int
                or prose_denominator < 1 or not 0 <= prose_numerator < prose_denominator):
            raise ValueError('Invalid integer prose fraction')
        expected = {'poetry', 'prose'} if prose_numerator else {'poetry'}
        if set(datasets) != expected:
            raise ValueError('Training domains do not match mixture')
        self.numerator, self.denominator = prose_numerator, prose_denominator
        self.total_targets = 0 if state is None else state['total_targets']
        if state is not None and (state['prose_fraction'] != [prose_numerator, prose_denominator]
                                  or set(state['domains']) != expected):
            raise ValueError('Resume mixture/domains changed')
        self.streams = {domain: EpochStream(datasets[domain], seed + (domain == 'prose'),
            None if state is None else state['domains'][domain]) for domain in ('poetry', 'prose') if domain in datasets}
        if type(self.total_targets) is not int or self.total_targets < 0:
            raise ValueError('Invalid total target count')
        self._check_counts()

    def _check_counts(self):
        prose = self.streams['prose'].total_targets if 'prose' in self.streams else 0
        if (prose != self.total_targets * self.numerator // self.denominator
                or sum(s.total_targets for s in self.streams.values()) != self.total_targets):
            raise ValueError('Domain target counts disagree with cumulative quotas')

    def state(self):
        return dict(total_targets=self.total_targets, prose_fraction=[self.numerator, self.denominator],
                    domains={name: stream.state() for name, stream in self.streams.items()})

    def statistics(self):
        result = dict(total_targets=self.total_targets, prose_fraction=[self.numerator, self.denominator], domains={})
        for domain in ('poetry', 'prose'):
            if domain not in self.streams:
                result['domains'][domain] = dict(targets=0, exposures=0., active=False, targets_per_epoch=None)
                continue
            stats = self.streams[domain].statistics()
            result['domains'][domain] = dict(**stats, active=True,
                exposures=stats['targets'] / stats['targets_per_epoch'])
        return result

    def next(self, quota, batch_size):
        if type(quota) is not int or quota < 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError('Positive integer quota and batch size required')
        before = self.total_targets * self.numerator // self.denominator
        prose = (self.total_targets + quota) * self.numerator // self.denominator - before
        counts = {'poetry': quota - prose, 'prose': prose}
        batches = []
        for domain, count in counts.items():
            if count:
                batches.extend(self.streams[domain].next(count, batch_size))
        if sum(int(labels[:, 1:].ne(-100).sum()) for _, labels in batches) != quota:
            raise AssertionError('Supervised targets differ from quota')
        self.total_targets += quota
        self._check_counts()
        return batches, counts


def capture_rng(device):
    # The shared helper probes all available CUDA devices even on CPU. Keep
    # local CPU fixtures off CUDA entirely; the CUDA path records its RNG too.
    values = np.random.get_state()
    return dict(python=random.getstate(), numpy=[values[0], values[1].tolist(), *values[2:]],
        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])


def input_identity(cfg, parent_hash, tokenizer_sha, training, validation, device, precision):
    files = {}
    for data in [*training.values(), *validation.values()]:
        files.update({str(data.folder / name): digest for name, digest in data.hashes.items()})
    return dict(config=cfg, parent_sha256=parent_hash, tokenizer_sha256=tokenizer_sha,
        input_freeze_sha256=validate_input_freeze(cfg, tokenizer_sha, training, validation),
        prepared_manifests_sha256={d: sha256(Path(cfg[d + '_data']) / 'manifest.json') for d in ('poetry', 'prose')},
        inputs_sha256=files, code_sha256=code_hashes(), trainer_sha256=sha256(__file__),
        dependencies_sha256={p: sha256(ROOT / p) for p in DEPENDENCIES},
        environment={**environment(device, precision), 'tokenizers': version('tokenizers')},
        best_criterion='minimum poetry validation token-weighted NLL; not a generation-quality gate')


def train(raw, run_dir, until_tokens, device_name='cpu', resume=False, stop_after=None):
    cfg = config(raw)
    if type(until_tokens) is not int or until_tokens not in cfg['milestones']:
        raise ValueError('until_tokens must be a declared milestone')
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError('stop_after must be a positive absolute optimizer-step boundary')
    if device_name not in ('cpu', 'cuda'):
        raise ValueError('Select cpu or cuda explicitly')
    run_dir = Path(run_dir)
    with run_lock(ROOT / 'runs/pretrain_series01', 'series'), run_lock(run_dir, 'training'):
        return train_locked(cfg, run_dir, until_tokens, device_name, resume, stop_after)


def train_locked(cfg, run_dir, until_tokens, device_name, resume, stop_after):
    started = time.monotonic()
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError('Refuse to overwrite run: ' + str(run_dir))
    if device_name == 'cuda':
        check_gpu_idle()
    torch.set_num_threads(cfg['cpu_threads'])
    device, precision = resolve_device(device_name, cfg['precision'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.random.default_generator.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])
    tokenizer_sha = sha256(cfg['tokenizer'])
    # Resume uses the checkpoint's persisted parent identity, not the parent
    # file. The selected parent may have moved to archival storage by then.
    state = load_checkpoint(run_dir / 'last.pt' if resume else cfg['parent'])
    validate_checkpoint(state, cfg, tokenizer_sha, initial=not resume)
    parent_hash = state['identity']['parent_sha256'] if resume else sha256(cfg['parent'])
    training, validation = load_datasets(cfg, tokenizer_sha)
    identity = input_identity(cfg, parent_hash, tokenizer_sha, training, validation, device, precision)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if resume and (state['fingerprint'] != fingerprint or state['identity'] != identity):
        raise ValueError('Exact resume requires same config, data, code and environment')
    model = GPT(ModelConfig(**cfg['model'])).to(device)
    model.load_state_dict(state['model'])
    opts = optimizers(model, cfg, device)
    lineage = deepcopy(state['lineage'])
    step, tokens_seen, previous_seconds, attempt_id, best_nll = 0, 0, 0., 1, None
    if resume:
        restore_optimizers(opts, state, cfg['optimizer'])
        step, tokens_seen, best_nll = state['step'], state['tokens_seen'], state['best_nll']
        previous_seconds, attempt_id = resume_accounting(run_dir, fingerprint, state)
        parent_steps, parent_tokens = state['parent_steps'], state['parent_tokens_seen']
    else:
        parent_steps = state.get('parent_steps', 0) + state['step']
        parent_tokens = state.get('parent_tokens_seen', 0) + state['tokens_seen']
        lineage['parents'].append(dict(checkpoint_sha256=parent_hash, domain=state['config']['domain'],
            step=state['step'], tokens_seen=state['tokens_seen'], stage='poetry08',
            optimizer='new AdamW' if cfg['optimizer'] == 'adamw' else 'new Muon + AdamW',
            parent_optimizer=state['config'].get('optimizer'), optimizer_state='fresh'))
    stream = MixtureStream(training, cfg['seed'], 0 if cfg['mode'] == 'pure' else 1,
                          state=state['streams'] if resume else None)
    if stream.total_targets != tokens_seen or tokens_seen > until_tokens:
        raise ValueError('Stream/checkpoint count mismatch or milestone precedes saved checkpoint')
    if resume:
        restore_rng(state['rng'])
    del state
    model.train()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'config.json', cfg)
    write_json(run_dir / 'environment.json', identity)
    metrics = run_dir / 'metrics.jsonl'
    repair_incomplete_jsonl_tail(metrics)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    def elapsed():
        return previous_seconds + time.monotonic() - started

    def event(name, **extra):
        append_event(metrics, dict(event=name, step=step, fingerprint=fingerprint,
            attempt_id=attempt_id, consumed_seconds=elapsed(), **extra))

    def status(name, **extra):
        write_json(run_dir / 'status.json', dict(status=name, step=step, fingerprint=fingerprint,
            attempt_id=attempt_id, consumed_seconds=elapsed(), supervised_tokens_seen=tokens_seen,
            comparison_budget=until_tokens, schedule_tokens=cfg['schedule_tokens'],
            stream=stream.statistics(), **extra))

    def save(name='last.pt'):
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Refuse to save nonfinite weights')
        save_checkpoint(run_dir / name, dict(format='poetrygpt-v1', config=cfg,
            model_config=cfg['model'], model=model.state_dict(),
            **optimizer_checkpoint(opts, cfg['optimizer']),
            step=step, tokens_seen=tokens_seen, streams=stream.state(), rng=capture_rng(device),
            best_nll=best_nll, best_criterion=identity['best_criterion'], lineage=lineage,
            parent_steps=parent_steps, parent_tokens_seen=parent_tokens,
            identity=identity, fingerprint=fingerprint, tokenizer_sha256=tokenizer_sha,
            prepared_manifest_sha256=identity['prepared_manifests_sha256']['poetry'],
            prepared_manifests_sha256=identity['prepared_manifests_sha256'],
            consumed_seconds=elapsed(), attempt_id=attempt_id))

    event('resume' if resume else 'start', parameters=model.parameter_count(),
          rollback_after_step=step if resume else None, comparison_budget=until_tokens)
    if not resume:
        save()
    status('running')
    try:
        while tokens_seen < until_tokens:
            if ((run_dir / 'STOP').exists() or elapsed() >= cfg['max_active_seconds']
                    or (stop_after is not None and step >= stop_after)):
                save()
                status('stopped_at_update_boundary')
                return run_dir / 'last.pt'
            update_started = time.perf_counter()
            count = update_quota(tokens_seen, cfg)
            batches, counts = stream.next(count, cfg['batch_size'])
            lr = token_lr(tokens_seen + count, cfg)
            for opt in opts.values():
                for group in opt.param_groups:
                    group['lr'] = lr
                opt.zero_grad(set_to_none=True)
            loss = backward_microbatches(model, batches, count, device, precision)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            for opt in opts.values():
                opt.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            step += 1
            tokens_seen += count
            at_milestone = tokens_seen in cfg['milestones']
            seconds = time.perf_counter() - update_started
            event('train', nll=loss, lr=lr, supervised_targets=count, domain_targets=counts,
                total_supervised_tokens=tokens_seen, stream=stream.statistics(), seconds=seconds,
                targets_per_second=count / seconds, grad_norm_before_clip=float(norm),
                peak_cuda_allocated_mib=(torch.cuda.max_memory_allocated(device) / 1024**2
                                         if device.type == 'cuda' else 0.))
            if step % cfg['eval_interval'] == 0 or at_milestone:
                saved_rng = capture_rng(device)
                try:
                    reports = {d: evaluate(model, data, device, precision, cfg['eval_batch_size'], cfg['eval_batches'])
                               for d, data in validation.items()}
                finally:
                    restore_rng(saved_rng)
                write_json(run_dir / 'validation_history' / f'step_{step:08d}.json',
                           dict(step=step, tokens_seen=tokens_seen, domains=reports))
                event('validation', **reports['poetry']['metrics']['overall'],
                      domains={d: report['metrics']['overall'] for d, report in reports.items()})
                score = reports['poetry']['metrics']['overall']['nll']
                if best_nll is None or score < best_nll:
                    best_nll = score
                    save('best.pt')
            if step % cfg['save_interval'] == 0 or at_milestone:
                save()
            if step % cfg['log_interval'] == 0 or at_milestone:
                status('running', last_train_nll=loss)
                print(f'step={step} targets={tokens_seen} nll={loss:.4f}', flush=True)
        status('comparison_budget_reached')
    except BaseException as error:
        # Advanced streams and partially updated parameters must never replace
        # the last complete update. Resume explicitly rolls back the log tail.
        event('failure', error=repr(error))
        status('failed_or_interrupted', error=repr(error), recovery='Resume last.pt; replay unsaved updates')
        raise
    return run_dir / 'last.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--until-tokens', type=int, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    train(read_json(args.config), args.run_dir, args.until_tokens, args.device, args.resume, args.stop_after)


if __name__ == '__main__':
    main()
