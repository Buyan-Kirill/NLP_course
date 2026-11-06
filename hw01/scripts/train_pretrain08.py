"""From-scratch A/B/C pretraining with token budgets and exact epoch-phase resume.

Historical trainers remain unchanged. Reuse their checked serialization/math,
but regenerate within-work windows at each epoch and retain every quota tail.
Stopping at a comparison budget does not restart the 1.05B-token LR schedule.
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
import numpy as np
import torch
from tokenizers import Tokenizer

from poetrygpt.data import BatchCursor, PreparedDataset, read_json, sha256
from poetrygpt.evaluate import evaluate
from poetrygpt.model import GPT, ModelConfig
from poetrygpt.runtime import (append_event, code_hashes, environment, load_checkpoint,
    repair_incomplete_jsonl_tail, resolve_device, restore_rng, rng_state, run_lock,
    save_checkpoint, seed_everything, write_json)
from poetrygpt.train import backward_microbatches, make_optimizer
from scripts.train_causal_series import check_gpu_idle, padded
from scripts.train_narrative_series04 import scene_windows, resume_accounting


DEFAULTS = dict(data='data/prepared_v7/expanded',
    tokenizer='artifacts/tokenizers_v3/selected/tokenizer.json', seed=2026110308,
    optimizer='adamw', target_tokens=16384, batch_size=32,
    schedule_tokens=1050000000, warmup_tokens=10000000,
    lr=3e-4, min_lr=3e-5, weight_decay=.1, betas=[.9, .95], eps=1e-8,
    grad_clip=1., precision='bf16', cpu_threads=4, fused_optimizer=True,
    save_interval=1000, eval_interval=2000, eval_batches=32, eval_batch_size=16,
    log_interval=100, max_active_seconds=21600)


def config(raw):
    if set(raw) - DEFAULTS.keys() - {'model'}:
        raise ValueError('Unknown pretrain08 config fields')
    cfg = {**deepcopy(DEFAULTS), **raw}
    cfg['model'] = ModelConfig(**cfg.get('model', {})).__dict__.copy()
    for key in ('seed', 'target_tokens', 'batch_size', 'schedule_tokens', 'warmup_tokens',
                'cpu_threads', 'save_interval', 'eval_interval', 'eval_batches',
                'eval_batch_size', 'log_interval', 'max_active_seconds'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError('Expected positive integer: ' + key)
    if cfg['warmup_tokens'] >= cfg['schedule_tokens'] or cfg['seed'] >= 2**32:
        raise ValueError('Invalid warmup or seed')
    for key in ('lr', 'min_lr', 'weight_decay', 'eps', 'grad_clip'):
        if type(cfg[key]) not in (float, int) or not math.isfinite(cfg[key]):
            raise ValueError('Expected finite scalar: ' + key)
    if not 0 < cfg['min_lr'] <= cfg['lr'] or cfg['weight_decay'] < 0 or min(cfg['eps'], cfg['grad_clip']) <= 0:
        raise ValueError('Invalid optimizer scalars')
    if len(cfg['betas']) != 2 or any(not 0 <= b < 1 for b in cfg['betas']):
        raise ValueError('Invalid AdamW betas')
    if cfg['optimizer'] not in ('adamw', 'muon') or cfg['precision'] not in ('fp32', 'bf16'):
        raise ValueError('Unsupported optimizer/precision')
    return cfg


def token_lr(end_tokens, cfg):
    if end_tokens <= cfg['warmup_tokens']:
        return cfg['lr'] * end_tokens / cfg['warmup_tokens']
    progress = min(1., (end_tokens - cfg['warmup_tokens']) /
                   (cfg['schedule_tokens'] - cfg['warmup_tokens']))
    return cfg['min_lr'] + .5 * (cfg['lr'] - cfg['min_lr']) * (1 + math.cos(math.pi * progress))


class EpochStream:
    """Same complete target set each epoch, changing context cuts and shuffle."""
    def __init__(self, data, seed, state=None):
        self.data, self.seed = data, seed
        self.total_targets = 0
        self.record_targets = np.zeros(len(data.metadata), dtype=np.int64)
        self.pending = None
        self._set_epoch(0 if state is None else state['cursor']['epoch'])
        if state is not None:
            self.cursor = BatchCursor(len(self.windows), 1, seed, **state['cursor'])
            self.pending = deepcopy(state['pending'])
            self.total_targets = state['total_targets']
            self.record_targets = np.asarray(state['record_targets'], dtype=np.int64)
            if self.record_targets.shape != (len(data.metadata),) or self.record_targets.sum() != self.total_targets:
                raise ValueError('Invalid stream coverage state')

    def _set_epoch(self, epoch):
        special = self.data.manifest['special_tokens']
        self.windows = np.asarray(scene_windows(self.data.ids, self.data.metadata,
            self.data.context_length, special['<|eos|>'], self.seed + epoch,
            initial_token=int(self.data.ids[0])), dtype=np.int64)
        self.cursor = BatchCursor(len(self.windows), 1, self.seed, epoch=epoch)

    def state(self):
        return dict(cursor=self.cursor.state_dict(), pending=deepcopy(self.pending),
                    total_targets=self.total_targets, record_targets=self.record_targets.tolist())

    def statistics(self):
        return dict(targets=self.total_targets, epoch=self.cursor.epoch,
            epoch_offset=self.cursor.offset, windows_in_epoch=len(self.windows),
            targets_per_epoch=len(self.data.ids) - 1,
            records_seen=int(np.count_nonzero(self.record_targets)), pending=deepcopy(self.pending))

    def next(self, quota, batch_size):
        if type(quota) is not int or quota < 1 or batch_size < 1:
            raise ValueError('Positive quota and batch size required')
        records = []
        while quota:
            if self.pending is None:
                if self.cursor.offset == self.cursor.size:
                    self._set_epoch(self.cursor.epoch + 1)
                self.pending = dict(index=self.cursor.next()[0], consumed=0)
            start, end, owner = map(int, self.windows[self.pending['index']])
            first = start + 1 + self.pending['consumed']
            stop = min(end, first + quota)
            if stop <= first:
                raise ValueError('Invalid pending window')
            ids = np.asarray(self.data.ids[start:stop], dtype=np.int64).tolist()
            labels = [-100] * (first - start) + ids[first - start:]
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


def optimizers(model, cfg, device):
    if cfg['optimizer'] == 'adamw':
        return {'adamw': make_optimizer(model, cfg, device)}
    # Tied embedding/head is one Parameter and must be updated exactly once.
    embedding_ids = {id(p) for module in model.modules() if isinstance(module, torch.nn.Embedding)
                     for p in module.parameters()}
    hidden = [p for p in model.parameters() if p.ndim == 2 and id(p) not in embedding_ids]
    other = [p for p in model.parameters() if p.ndim != 2 or id(p) in embedding_ids]
    if len({id(p) for p in hidden + other}) != len(list(model.parameters())):
        raise ValueError('Optimizer parameter coverage mismatch')
    return {
        'muon': torch.optim.Muon(hidden, lr=cfg['lr'], weight_decay=cfg['weight_decay'],
            momentum=.95, ns_steps=5, adjust_lr_fn='match_rms_adamw'),
        'adamw': torch.optim.AdamW([
            dict(params=[p for p in other if p.ndim >= 2], weight_decay=cfg['weight_decay']),
            dict(params=[p for p in other if p.ndim < 2], weight_decay=0.)],
            lr=cfg['lr'], betas=tuple(cfg['betas']), eps=cfg['eps'],
            fused=bool(cfg['fused_optimizer'] and device.type == 'cuda'))}


def train(raw, run_dir, until_tokens, device_name='cuda', resume=False, stop_after=None):
    cfg = config(raw)
    if type(until_tokens) is not int or not 0 < until_tokens <= cfg['schedule_tokens']:
        raise ValueError('Invalid comparison token budget')
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError('Invalid stop_after')
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
    seed_everything(cfg['seed'])
    train_data = PreparedDataset(cfg['data'], 'prose', 'train', cfg['model']['context_length'])
    validation = PreparedDataset(cfg['data'], 'prose', 'validation', cfg['model']['context_length'])
    tokenizer_sha = sha256(cfg['tokenizer'])
    declared = train_data.manifest['inputs_sha256']
    if tokenizer_sha not in [v for k, v in declared.items() if k.replace('\\', '/').endswith('tokenizer.json')]:
        raise ValueError('Tokenizer differs from frozen prepared data')
    if Tokenizer.from_file(cfg['tokenizer']).get_vocab_size() != cfg['model']['vocab_size']:
        raise ValueError('Model/tokenizer vocabulary mismatch')
    identity = dict(config=cfg, prepared_manifest_sha256=sha256(Path(cfg['data']) / 'manifest.json'),
        tokenizer_sha256=tokenizer_sha, inputs_sha256={**train_data.hashes, **validation.hashes},
        code_sha256=code_hashes(), trainer_sha256=sha256(__file__),
        helpers_sha256={name: sha256(ROOT / 'scripts' / name) for name in
            ('train_narrative_series04.py', 'train_causal_series.py')},
        environment=environment(device, precision), evaluation='fixed packed validation subset; same for A/B/C')
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    model = GPT(ModelConfig(**cfg['model'])).to(device)
    opts = optimizers(model, cfg, device)
    step, tokens_seen, previous_seconds, attempt_id, best_nll = 0, 0, 0., 1, None
    lineage = dict(origin='random_initialization', seed=cfg['seed'], parents=[])
    stream_state, saved_rng = None, None
    if resume:
        state = load_checkpoint(run_dir / 'last.pt')
        if state['fingerprint'] != fingerprint or set(state['optimizers']) != set(opts):
            raise ValueError('Exact resume requires same config, inputs, code and environment')
        model.load_state_dict(state['model'])
        for name, opt in opts.items():
            opt.load_state_dict(state['optimizers'][name])
        step, tokens_seen, best_nll = state['step'], state['tokens_seen'], state['best_nll']
        previous_seconds, attempt_id = resume_accounting(run_dir, fingerprint, state)
        stream_state, saved_rng = state['stream'], state['rng']
        del state
    if tokens_seen > until_tokens:
        raise ValueError('Budget precedes saved checkpoint')
    stream = EpochStream(train_data, cfg['seed'], stream_state)
    if stream.total_targets != tokens_seen:
        raise ValueError('Stream/checkpoint token count mismatch')
    if saved_rng is not None:
        restore_rng(saved_rng)
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
        append_event(metrics, dict(event=name, step=step, attempt_id=attempt_id,
            fingerprint=fingerprint, consumed_seconds=elapsed(), **extra))

    def status(name, **extra):
        write_json(run_dir / 'status.json', dict(status=name, step=step, fingerprint=fingerprint,
            attempt_id=attempt_id, consumed_seconds=elapsed(), supervised_tokens_seen=tokens_seen,
            comparison_budget=until_tokens, schedule_tokens=cfg['schedule_tokens'],
            stream=stream.statistics(), **extra))

    def save(name='last.pt'):
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Refuse to replace checkpoint with nonfinite weights')
        save_checkpoint(run_dir / name, dict(format='poetrygpt-v1', model_config=cfg['model'],
            config={**cfg, 'domain': 'prose'}, model=model.state_dict(),
            optimizers={key: opt.state_dict() for key, opt in opts.items()},
            step=step, tokens_seen=tokens_seen, stream=stream.state(), rng=rng_state(),
            best_nll=best_nll, lineage=lineage, fingerprint=fingerprint,
            tokenizer_sha256=tokenizer_sha, prepared_manifest_sha256=identity['prepared_manifest_sha256'],
            consumed_seconds=elapsed(), attempt_id=attempt_id))

    event('resume' if resume else 'start', parameters=model.parameter_count(),
          rollback_after_step=step if resume else None, comparison_budget=until_tokens)
    model.train()
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
            update_start = time.perf_counter()
            count = min(cfg['target_tokens'], until_tokens - tokens_seen)
            batches = stream.next(count, cfg['batch_size'])
            if sum(int(labels[:, 1:].ne(-100).sum()) for _, labels in batches) != count:
                raise AssertionError('Supervision quota mismatch')
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
            seconds = time.perf_counter() - update_start
            event('train', nll=loss, lr=lr, supervised_targets=count, total_supervised_tokens=tokens_seen,
                seconds=seconds, targets_per_second=count / seconds, epoch=stream.cursor.epoch,
                grad_norm_before_clip=float(norm), peak_cuda_allocated_mib=(
                    torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else None))
            if step % cfg['eval_interval'] == 0 or tokens_seen == until_tokens:
                report = evaluate(model, validation, device, precision, cfg['eval_batch_size'], cfg['eval_batches'])
                score = report['metrics']['overall']['nll']
                write_json(run_dir / 'validation_history' / f'step_{step:08d}.json', dict(step=step, **report))
                event('validation', **report['metrics']['overall'])
                if best_nll is None or score < best_nll:
                    best_nll = score
                    save('best.pt')
            if step % cfg['save_interval'] == 0 or tokens_seen == until_tokens:
                save()
            if step % cfg['log_interval'] == 0 or tokens_seen == until_tokens:
                status('running', last_train_nll=loss)
                print(f'step={step} targets={tokens_seen} nll={loss:.4f} seconds={elapsed():.1f}', flush=True)
        status('comparison_budget_reached')
    except BaseException as error:
        # Never save weights or advanced sampler state from a half-finished update.
        event('failure', error=repr(error))
        status('failed_or_interrupted', error=repr(error), recovery='Resume last.pt; replay unsaved updates')
        raise
    return run_dir / 'last.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--until-tokens', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    train(read_json(args.config), args.run_dir, args.until_tokens, args.device, args.resume, args.stop_after)


if __name__ == '__main__':
    main()
