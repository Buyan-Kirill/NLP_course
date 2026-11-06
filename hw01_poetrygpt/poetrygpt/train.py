"""Single-GPU/CPU pretraining and SFT, starting from random project weights."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .data import BatchCursor, PreparedDataset, read_json, sha256
from .evaluate import evaluate
from .model import GPT, ModelConfig, token_losses
from .runtime import (append_event, autocast, code_hashes, environment, load_checkpoint,
                      locked_run, repair_incomplete_jsonl_tail, resolve_device, restore_rng,
                      rng_state, save_checkpoint, seed_everything, write_json)


DEFAULTS = {'data': 'data/prepared_v3', 'tokenizer': 'artifacts/tokenizers_v3/selected/tokenizer.json',
            'seed': 20261031, 'batch_size': 8, 'accumulation_steps': 4, 'max_steps': 1000,
            'lr': 3e-4, 'min_lr': 3e-5, 'warmup_steps': 50, 'weight_decay': 0.1,
            'betas': [0.9, 0.95], 'eps': 1e-8, 'grad_clip': 1.0, 'precision': 'auto',
            'eval_interval': 100, 'eval_batches': 32, 'save_interval': 50,
            'cpu_threads': 4, 'deterministic': False, 'fused_optimizer': True,
            'early_stopping_patience': 0, 'min_delta': 0.0}

WSD_KEYS = {'rewarm_steps', 'decay_steps', 'initial_lr'}


def validated_config(raw):
    unknown = set(raw) - set(DEFAULTS) - {'domain', 'model', 'lr_schedule'} - WSD_KEYS
    if unknown:
        raise ValueError(f'Unknown config keys: {sorted(unknown)}')
    cfg = {**deepcopy(DEFAULTS), **raw}
    cfg['model'] = ModelConfig(**cfg.get('model', {})).__dict__.copy()
    if cfg.get('domain') not in ('prose', 'poetry', 'sft'):
        raise ValueError('Select prose, poetry or sft')
    for field in ('batch_size', 'accumulation_steps', 'max_steps', 'eval_interval', 'save_interval', 'cpu_threads'):
        if type(cfg[field]) is not int or cfg[field] < 1:
            raise ValueError(f'{field} must be a positive integer')
    for field in ('warmup_steps', 'eval_batches', 'early_stopping_patience'):
        if type(cfg[field]) is not int or cfg[field] < 0:
            raise ValueError(f'{field} must be nonnegative')
    if not 0 <= cfg['warmup_steps'] < cfg['max_steps']:
        raise ValueError('Warmup must be shorter than run')
    if any(not math.isfinite(cfg[k]) for k in ('lr', 'min_lr', 'grad_clip', 'eps', 'weight_decay', 'min_delta')):
        raise ValueError('Optimizer values must be finite')
    if not 0 < cfg['min_lr'] <= cfg['lr'] or cfg['grad_clip'] <= 0 or cfg['eps'] <= 0 or cfg['weight_decay'] < 0 or cfg['min_delta'] < 0:
        raise ValueError('Invalid optimizer/schedule values')
    if len(cfg['betas']) != 2 or any(not 0 <= b < 1 for b in cfg['betas']):
        raise ValueError('Invalid Adam betas')
    schedule = cfg.get('lr_schedule', 'cosine')
    if schedule not in ('cosine', 'wsd'):
        raise ValueError('lr_schedule must be cosine or wsd')
    if schedule == 'wsd':
        if WSD_KEYS - cfg.keys():
            raise ValueError('WSD requires rewarm_steps, decay_steps and initial_lr')
        if cfg['warmup_steps'] != 0:
            raise ValueError('WSD requires warmup_steps=0; use rewarm_steps')
        if type(cfg['rewarm_steps']) is not int or cfg['rewarm_steps'] < 0:
            raise ValueError('WSD rewarm_steps must be a nonnegative integer')
        if type(cfg['decay_steps']) is not int or cfg['decay_steps'] < 2:
            raise ValueError('WSD decay_steps must be an integer >= 2 to include both endpoints')
        if cfg['rewarm_steps'] + cfg['decay_steps'] > cfg['max_steps']:
            raise ValueError('WSD rewarm and decay must fit within max_steps')
        initial = cfg['initial_lr']
        if (isinstance(initial, bool) or not isinstance(initial, (int, float))
                or not math.isfinite(initial) or not 0 < initial <= cfg['lr']):
            raise ValueError('WSD initial_lr must be finite, positive and <= lr')
        if cfg['rewarm_steps'] < 2 and initial != cfg['lr']:
            raise ValueError('Changing WSD initial_lr requires at least two rewarm updates')
    elif WSD_KEYS & cfg.keys():
        raise ValueError('WSD fields require lr_schedule=wsd')
    return cfg


def learning_rate(step, cfg):
    """step is the zero-based next optimizer update; first LR is nonzero."""
    if cfg.get('lr_schedule', 'cosine') == 'wsd':
        # Counts include both endpoints: rewarm[0]=initial, rewarm[-1]=peak;
        # decay[0]=peak, decay[-1]=min. The intervening updates keep peak LR.
        rewarm, decay = cfg['rewarm_steps'], cfg['decay_steps']
        if step < rewarm:
            if step == 0:
                return cfg['initial_lr']
            if step == rewarm - 1:
                return cfg['lr']
            return cfg['initial_lr'] + (cfg['lr'] - cfg['initial_lr']) * step / (rewarm - 1)
        start_decay = cfg['max_steps'] - decay
        if step <= start_decay:
            return cfg['lr']
        if step >= cfg['max_steps'] - 1:
            return cfg['min_lr']
        progress = (step - start_decay) / (decay - 1)
        return cfg['lr'] + (cfg['min_lr'] - cfg['lr']) * progress
    warmup, total = cfg['warmup_steps'], cfg['max_steps']
    if step < warmup:
        return cfg['lr'] * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup - 1))
    return cfg['min_lr'] + 0.5 * (cfg['lr'] - cfg['min_lr']) * (1 + math.cos(math.pi * progress))


def make_optimizer(model, cfg, device):
    matrices = [p for p in model.parameters() if p.ndim >= 2]
    vectors = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{'params': matrices, 'weight_decay': cfg['weight_decay']},
                              {'params': vectors, 'weight_decay': 0.0}],
                             lr=cfg['lr'], betas=tuple(cfg['betas']), eps=cfg['eps'],
                             fused=bool(cfg['fused_optimizer'] and device.type == 'cuda'))


def backward_microbatches(model, batches, denominator, device, precision):
    """Accumulate sum CE / total supervised targets, not mean of micro means."""
    if denominator <= 0:
        raise ValueError('Optimizer step has no supervised targets')
    loss_sum = 0.0
    for ids, labels in batches:
        ids, labels = ids.to(device), labels.to(device)
        with autocast(device, precision):
            losses, mask = token_losses(model(ids), labels)
            summed = losses.sum()
        if not torch.isfinite(summed):
            raise FloatingPointError('Nonfinite training loss; previous checkpoint retained')
        (summed / denominator).backward()
        loss_sum += summed.detach().item()
    return loss_sum / denominator


def validate_continuation(state, checkpoint, cfg, identity, model, optimizer, train_size, allow_dropout_change=False):
    """A new schedule on the same training stream; never relax exact resume.

    Old checkpoints keep their identity in the adjacent environment.json. Verify
    that original identity before comparing it with the new run. Only train.py
    may change here, and both code hashes are retained in the child's lineage.
    """
    parent_identity = read_json(Path(checkpoint).parent / 'environment.json')
    digest = hashlib.sha256(json.dumps(parent_identity, sort_keys=True).encode()).hexdigest()
    if digest != state['fingerprint'] or parent_identity['config'] != state['config']:
        raise ValueError('Continuation parent identity does not match checkpoint')
    allowed = {'lr', 'min_lr', 'warmup_steps', 'max_steps', 'eval_interval', 'eval_batches',
               'save_interval', 'early_stopping_patience', 'min_delta', 'lr_schedule'} | WSD_KEYS
    changed = {key for key in set(cfg) | set(state['config']) if cfg.get(key) != state['config'].get(key)}
    if allow_dropout_change:
        parent_model = ModelConfig(**state['config']['model']).__dict__.copy()
        parent_model['dropout'] = cfg['model']['dropout']
        if parent_model != cfg['model']:
            raise ValueError('Dropout continuation cannot change other model fields')
        changed.discard('model')
    if changed - allowed:
        raise ValueError(f'Continuation cannot change training semantics: {sorted(changed - allowed)}')
    for key in ('prepared_manifest_sha256', 'tokenizer_sha256', 'inputs_sha256', 'environment'):
        if parent_identity[key] != identity[key]:
            raise ValueError(f'Continuation {key} differs from parent')
    if state['prepared_manifest_sha256'] != identity['prepared_manifest_sha256']:
        raise ValueError('Continuation checkpoint prepared manifest mismatch')
    old_code, new_code = parent_identity['code_sha256'], identity['code_sha256']
    changed_code = {key for key in set(old_code) | set(new_code) if old_code.get(key) != new_code.get(key)}
    if changed_code - {'train.py'}:
        raise ValueError(f'Continuation requires unchanged model/data/runtime code: {sorted(changed_code)}')
    parent_steps = state.get('parent_steps', 0) + state['step']
    if type(parent_steps) is not int or parent_steps < 1:
        raise ValueError('Continuation requires a trained optimizer checkpoint')
    # AdamW's parameter numbering is positional, so validate each group and
    # moment shape before load_state_dict can silently accept a malformed state.
    saved = state['optimizer']
    if len(saved['param_groups']) != len(optimizer.param_groups):
        raise ValueError('Continuation optimizer group mismatch')
    expected_ids = set()
    for old_group, new_group in zip(saved['param_groups'], optimizer.param_groups):
        if len(old_group['params']) != len(new_group['params']):
            raise ValueError('Continuation optimizer parameter count mismatch')
        for key in ('weight_decay', 'betas', 'eps', 'fused', 'amsgrad', 'maximize'):
            if old_group.get(key) != new_group.get(key):
                raise ValueError(f'Continuation optimizer option mismatch: {key}')
        for identifier, parameter in zip(old_group['params'], new_group['params']):
            expected_ids.add(identifier)
            moment = saved['state'].get(identifier, {})
            for key in ('exp_avg', 'exp_avg_sq'):
                value = moment.get(key)
                if not torch.is_tensor(value) or value.shape != parameter.shape or not torch.isfinite(value).all():
                    raise ValueError(f'Continuation invalid optimizer moment: {key}')
            update = moment.get('step')
            if not torch.is_tensor(update) or update.numel() != 1 or update.item() != parent_steps:
                raise ValueError('Continuation optimizer step mismatch')
    if expected_ids != set(saved['state']):
        raise ValueError('Continuation optimizer state coverage mismatch')
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Continuation nonfinite model weights')
    microsteps = parent_steps * cfg['accumulation_steps']
    batches_per_epoch = math.ceil(train_size / cfg['batch_size'])
    expected_cursor = {'epoch': (microsteps - 1) // batches_per_epoch,
                       'offset': min(((microsteps - 1) % batches_per_epoch + 1) * cfg['batch_size'], train_size)}
    if state['cursor'] != expected_cursor:
        raise ValueError('Continuation cursor does not match completed updates')
    return parent_identity, parent_steps


@locked_run('run_dir', 'training')
def train(raw_config, run_dir, device_name='auto', resume=None, init_from=None, stop_after=None, continue_from=None,
          allow_dropout_change=False):
    cfg = validated_config(raw_config)
    if sum(bool(path) for path in (resume, init_from, continue_from)) > 1:
        raise ValueError('Choose one of resume, init-from or continue-from')
    if allow_dropout_change and not continue_from:
        raise ValueError('allow-dropout-change requires continue-from; exact resume never changes config')
    if cfg['domain'] == 'sft' and not (resume or init_from or continue_from):
        raise ValueError('SFT requires our own pretrained checkpoint via --init-from')
    if stop_after is not None and stop_after < 1:
        raise ValueError('stop-after must be positive')
    run_dir = Path(run_dir)
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError('New run directory must be empty; refuse overwriting previous results')
    if resume and (not run_dir.is_dir() or Path(resume).resolve().parent != run_dir.resolve()):
        raise ValueError('Resume checkpoint must belong to the selected run directory')
    device, precision = resolve_device(device_name, cfg['precision'])
    torch.set_num_threads(cfg['cpu_threads'])
    torch.use_deterministic_algorithms(cfg['deterministic'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    seed_everything(cfg['seed'])
    train_data = PreparedDataset(cfg['data'], cfg['domain'], 'train', cfg['model']['context_length'])
    validation = PreparedDataset(cfg['data'], cfg['domain'], 'validation', cfg['model']['context_length'])
    tokenizer_sha = sha256(cfg['tokenizer'])
    declared = train_data.manifest['inputs_sha256']
    expected = [v for k, v in declared.items() if k.replace('\\', '/').endswith('tokenizers_v3/selected/tokenizer.json')]
    if expected != [tokenizer_sha]:
        raise ValueError('Tokenizer does not match the prepared release')
    tokenizer_json = read_json(cfg['tokenizer'])
    if len(tokenizer_json['model']['vocab']) != cfg['model']['vocab_size']:
        raise ValueError('Model vocabulary differs from tokenizer')
    manifest_sha = sha256(Path(cfg['data']) / 'manifest.json')
    identity = {'config': cfg, 'prepared_manifest_sha256': manifest_sha, 'tokenizer_sha256': tokenizer_sha,
                'inputs_sha256': {**train_data.hashes, **validation.hashes}, 'code_sha256': code_hashes(),
                'environment': environment(device, precision)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    model = GPT(ModelConfig(**cfg['model'])).to(device)
    optimizer = make_optimizer(model, cfg, device)
    cursor = BatchCursor(len(train_data), cfg['batch_size'], cfg['seed'])
    step, tokens_seen, best_nll, stale_evals = 0, 0, None, 0
    parent_steps, parent_tokens_seen = 0, 0
    lineage = {'origin': 'random_initialization', 'seed': cfg['seed'], 'parents': []}
    if resume or init_from or continue_from:
        state = load_checkpoint(resume or init_from or continue_from)
        parent_model_config = ModelConfig(**state['model_config']).__dict__.copy()
        compared_model_config = parent_model_config.copy()
        if allow_dropout_change:
            if parent_model_config['dropout'] == cfg['model']['dropout']:
                raise ValueError('allow-dropout-change requires an actual dropout change')
            compared_model_config['dropout'] = cfg['model']['dropout']
        if compared_model_config != cfg['model'] or state['tokenizer_sha256'] != tokenizer_sha:
            raise ValueError('Model config/tokenizer mismatch with checkpoint')
        model.load_state_dict(state['model'])
        lineage = deepcopy(state['lineage'])
        if resume:
            if state['fingerprint'] != fingerprint:
                raise ValueError('Exact resume requires identical config, data, code and runtime; use init-from for a new stage')
            optimizer.load_state_dict(state['optimizer'])
            cursor = BatchCursor(len(train_data), cfg['batch_size'], cfg['seed'], **state['cursor'])
            step, tokens_seen = state['step'], state['tokens_seen']
            best_nll, stale_evals = state['best_nll'], state['stale_evals']
            parent_steps, parent_tokens_seen = state.get('parent_steps', 0), state.get('parent_tokens_seen', 0)
            restore_rng(state['rng'])  # Restore AFTER model and optimizer construction.
        elif continue_from:
            parent_identity, parent_steps = validate_continuation(
                state, continue_from, cfg, identity, model, optimizer, len(train_data), allow_dropout_change)
            optimizer.load_state_dict(state['optimizer'])
            cursor = BatchCursor(len(train_data), cfg['batch_size'], cfg['seed'], **state['cursor'])
            parent_tokens_seen = state.get('parent_tokens_seen', 0) + state['tokens_seen']
            lineage['parents'].append({
                'kind': 'optimizer_stream_continuation', 'checkpoint_sha256': sha256(continue_from),
                'fingerprint': state['fingerprint'], 'domain': state['config']['domain'],
                'step': state['step'], 'total_optimizer_steps': parent_steps,
                'total_supervised_tokens': parent_tokens_seen, 'cursor': state['cursor'],
                'parent_code_sha256': parent_identity['code_sha256'], 'child_code_sha256': identity['code_sha256'],
                'schedule': f"new local-step {cfg.get('lr_schedule', 'cosine')} schedule; optimizer, sampler and RNG restored"})
            if allow_dropout_change:
                lineage['parents'][-1]['dropout_change'] = {
                    'old': parent_model_config['dropout'], 'new': cfg['model']['dropout'],
                    'policy': 'explicit dropout-only opt-in; parameter shapes and optimizer state unchanged'}
            restore_rng(state['rng'])
        else:
            lineage['parents'].append({'checkpoint_sha256': sha256(init_from), 'domain': state['config']['domain'], 'step': state['step']})
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'config.json', cfg)
    write_json(run_dir / 'environment.json', identity)
    metrics_file = run_dir / 'metrics.jsonl'
    if resume:
        repair_incomplete_jsonl_tail(metrics_file)
    append_event(metrics_file, {'event': 'resume' if resume else 'start', 'step': step, 'fingerprint': fingerprint,
                                'parameters': model.parameter_count(), 'lineage': lineage})

    def checkpoint_state():
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Refuse to replace a checkpoint with nonfinite weights')
        return {'format': 'poetrygpt-v1', 'config': cfg, 'model_config': cfg['model'], 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'step': step, 'cursor': cursor.state_dict(), 'rng': rng_state(),
                'tokens_seen': tokens_seen, 'best_nll': best_nll, 'stale_evals': stale_evals, 'lineage': lineage,
                'parent_steps': parent_steps, 'parent_tokens_seen': parent_tokens_seen,
                'fingerprint': fingerprint, 'tokenizer_sha256': tokenizer_sha, 'prepared_manifest_sha256': manifest_sha}

    def status(name, **extra):
        write_json(run_dir / 'status.json', {'status': name, 'step': step, 'max_steps': cfg['max_steps'],
                                            'supervised_tokens_seen': tokens_seen,
                                            'total_supervised_tokens_seen': parent_tokens_seen + tokens_seen, **extra})

    if not resume:
        save_checkpoint(run_dir / 'last.pt', checkpoint_state())
    model.train()
    status('running')
    try:
        # A process may die after saving the stopping checkpoint but before
        # writing its terminal status. Honor the restored decision immediately.
        if resume and cfg['early_stopping_patience'] and stale_evals >= cfg['early_stopping_patience']:
            status('early_stopped')
            return run_dir / 'last.pt'
        while step < cfg['max_steps']:
            if (run_dir / 'STOP').exists() or (stop_after is not None and step >= stop_after):
                save_checkpoint(run_dir / 'last.pt', checkpoint_state())
                status('stopped_at_update_boundary')
                return run_dir / 'last.pt'
            started = time.perf_counter()
            batches = [train_data.batch(cursor.next(), torch.device('cpu')) for _ in range(cfg['accumulation_steps'])]
            targets = sum(int(labels[:, 1:].ne(-100).sum()) for _, labels in batches)
            lr = learning_rate(step, cfg)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = backward_microbatches(model, batches, targets, device, precision)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            step += 1
            tokens_seen += targets
            elapsed = time.perf_counter() - started
            metrics = {'event': 'train', 'step': step, 'nll': loss, 'lr': lr, 'grad_norm_before_clip': float(norm),
                       'supervised_targets': targets, 'targets_per_second': targets / elapsed,
                       'seconds': elapsed, 'epoch': cursor.epoch, 'epoch_offset': cursor.offset,
                       'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else None}
            if device.type == 'cuda':
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                metrics.update(peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2,
                               cuda_free_mib=free_bytes / 1024**2, cuda_total_mib=total_bytes / 1024**2)
            append_event(metrics_file, metrics)
            should_stop = False
            if step % cfg['eval_interval'] == 0 or step == cfg['max_steps']:
                report = evaluate(model, validation, device, precision, cfg['batch_size'], cfg['eval_batches'])
                score = report['metrics']['overall']['nll']
                improved = best_nll is None or score < best_nll - cfg['min_delta']
                best_nll, stale_evals = (score, 0) if improved else (best_nll, stale_evals + 1)
                write_json(run_dir / 'validation_latest.json', {'step': step, **report})
                write_json(run_dir / 'validation_history' / f'step_{step:08d}.json', {
                    'step': step, 'supervised_tokens_seen': tokens_seen,
                    'coverage': report['coverage'], 'windows': report['windows'],
                    'overall': report['metrics']['overall'], 'per_record': report['per_record']})
                append_event(metrics_file, {'event': 'validation', 'step': step, 'coverage': report['coverage'],
                                            **report['metrics']['overall']})
                if improved:
                    save_checkpoint(run_dir / 'best.pt', checkpoint_state())
                should_stop = bool(cfg['early_stopping_patience'] and stale_evals >= cfg['early_stopping_patience'])
                print(f"step={step} train_nll={loss:.4f} validation_nll={score:.4f} ({report['coverage']})", flush=True)
            if step % cfg['save_interval'] == 0 or should_stop or step == cfg['max_steps']:
                save_checkpoint(run_dir / 'last.pt', checkpoint_state())
            status('running', last_train_nll=loss)
            if should_stop:
                status('early_stopped')
                return run_dir / 'last.pt'
        status('completed')
    except BaseException as error:
        # Never serialize possibly half-updated weights/optimizer after an
        # interruption inside backward/step. Keep the last atomic checkpoint.
        status('failed_or_interrupted', error=repr(error), recovery='Resume last.pt; unsaved updates are replayed')
        append_event(metrics_file, {'event': 'failure', 'step': step, 'error': repr(error)})
        raise
    return run_dir / 'last.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', type=Path)
    group.add_argument('--init-from', type=Path)
    group.add_argument('--continue-from', type=Path, help='New run/schedule preserving optimizer, sampler and RNG')
    parser.add_argument('--allow-dropout-change', action='store_true',
                        help='Explicit dropout-only experiment with continue-from; forbidden for resume/init-from')
    parser.add_argument('--stop-after', type=int, help='Stop after this absolute update number; preserve LR schedule for resume testing')
    args = parser.parse_args()
    train(read_json(args.config), args.run_dir, args.device, args.resume, args.init_from, args.stop_after,
          args.continue_from, args.allow_dropout_change)


if __name__ == '__main__':
    main()
