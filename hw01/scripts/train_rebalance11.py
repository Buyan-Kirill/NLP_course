"""Fork a prose checkpoint with an exact-quota selected-span mixture.

The frozen targeted10 trainer stays unchanged. This CE-only runner reuses its
schedule, optimizer and runtime contracts, replacing only the source stream.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer

from poetrygpt.data import PreparedDataset, read_json, sha256
from poetrygpt.evaluate import evaluate
from poetrygpt.model import GPT, ModelConfig
from poetrygpt.runtime import (append_event, code_hashes, environment, load_checkpoint,
    repair_incomplete_jsonl_tail, resolve_device, restore_rng, rng_state, run_lock,
    save_checkpoint, write_json)
from poetrygpt.train import backward_microbatches
from scripts.train_causal_series import check_gpu_idle
from scripts.train_narrative_series04 import resume_accounting
from scripts.train_pretrain08 import EpochStream, optimizers
from scripts import train_targeted10 as targeted
from scripts.rebalance11_stream import MixedStream, SpanStream, read_spans


def config(raw):
    fields = {'selected_spans', 'selected_fraction', 'selected_seed'}
    cfg = targeted.config({k: v for k, v in raw.items() if k not in fields})
    cfg.update(selected_spans=raw.get('selected_spans'),
               selected_fraction=raw.get('selected_fraction', .3),
               selected_seed=raw.get('selected_seed', 2026110411))
    if (cfg['mode'] != 'ce' or cfg['context_overlap'] != 0
            or cfg['schedule'] != 'inherit' or cfg['adam_beta2'] is not None
            or cfg['weight_decay_multiplier'] != 1):
        raise ValueError('Mixture pilot supports CE/inherited schedule and unchanged optimizer recipe')
    if (type(cfg['selected_fraction']) not in (int, float)
            or not 0 <= cfg['selected_fraction'] <= .3
            or type(cfg['selected_seed']) is not int or not 0 <= cfg['selected_seed'] < 2**32):
        raise ValueError('Use selected fraction in [0,.3] and uint32 seed')
    if cfg['selected_fraction'] and (not isinstance(cfg['selected_spans'], str) or not cfg['selected_spans']):
        raise ValueError('A positive fraction requires selected_spans JSONL')
    for value in (cfg['parent'], cfg['selected_spans']):
        if value is not None and not Path(value).resolve().is_relative_to(ROOT):
            raise ValueError('Checkpoint and selected spans must stay within this project')
    return cfg


def selected_manifest(cfg, recipe, spans):
    """Bind coordinates to the exact prepared array, offsets and tokenizer."""
    if spans is None:
        return None
    path = Path(cfg['selected_spans']).with_name('manifest.json')
    manifest = read_json(path)
    paths = [Path(recipe['data']) / name for name in
             ('prose_train.npy', 'prose_train_offsets.json', 'manifest.json')]
    paths.append(Path(recipe['tokenizer']))
    expected = {p.resolve().relative_to(ROOT).as_posix(): sha256(p) for p in paths}
    if (manifest.get('source_sha256') != expected
            or manifest.get('selection_sha256') != sha256(cfg['selected_spans'])
            or manifest.get('selected_targets') != sum(s['target_end'] - s['target_start'] for s in spans)):
        raise ValueError('Selected manifest does not match exact source data/tokenizer/spans')
    return dict(path=str(path), sha256=sha256(path))


def train(raw, run_dir, device_name='cuda', resume=False, stop_after=None):
    cfg, run_dir = config(raw), Path(run_dir).resolve()
    if not run_dir.is_relative_to(ROOT) or run_dir == ROOT:
        raise ValueError('Run directory must stay within this project')
    if stop_after is not None and (type(stop_after) is not int or stop_after < 0):
        raise ValueError('stop_after must be a nonnegative complete local update')
    with run_lock(ROOT / 'runs/pretrain_series01', 'series'), run_lock(run_dir, 'training'):
        return train_locked(cfg, run_dir, device_name, resume, stop_after)


def train_locked(cfg, run_dir, device_name, resume, stop_after):
    started = time.monotonic()
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError('Refuse to overwrite a mixture run')
    state = load_checkpoint(run_dir / 'last.pt' if resume else cfg['parent'])
    if resume:
        if state.get('pilot_config') != cfg or state.get('pilot_kind') != 'rebalance11':
            raise ValueError('Exact mixture resume requires unchanged configuration')
        parent_recipe, parent_record = state['parent_recipe'], state['parent_record']
    else:
        if (state['config'].get('domain') != 'prose' or not state.get('optimizers')
                or state.get('pilot_config') is not None):
            raise ValueError('Fork requires a full pretrain08 prose checkpoint')
        parent_recipe = deepcopy(state['config'])
        parent_directory = Path(cfg['parent']).parent
        if parent_directory.name == 'stages':
            parent_directory = parent_directory.parent
        environment_path = parent_directory / 'environment.json'
        parent_environment = read_json(environment_path)
        if (targeted.digest(parent_environment) != state['fingerprint']
                or {**parent_environment['config'], 'domain': 'prose'} != parent_recipe):
            raise ValueError('Parent environment/checkpoint fingerprint mismatch')
        parent_record = dict(path=cfg['parent'], sha256=sha256(cfg['parent']),
            fingerprint=state['fingerprint'], tokens=state['tokens_seen'], step=state['step'],
            environment_path=str(environment_path), environment_sha256=sha256(environment_path),
            optimizer_groups=targeted.optimizer_group_settings(state['optimizers']))
    recipe = deepcopy(parent_recipe)
    for key in ('target_tokens', 'batch_size'):
        if cfg[key] is not None:
            recipe[key] = cfg[key]
    torch.set_num_threads(cfg['cpu_threads'])
    device, precision = resolve_device(device_name, recipe['precision'])
    if device.type == 'cuda':
        check_gpu_idle()
        available = torch.cuda.get_device_properties(device).total_memory
        fraction = min(1., cfg['max_cuda_allocated_mib'] * 1024**2 / available)
        torch.cuda.set_per_process_memory_fraction(fraction, device.index or torch.cuda.current_device())
        torch.cuda.reset_peak_memory_stats(device)
    training = PreparedDataset(recipe['data'], 'prose', 'train', recipe['model']['context_length'])
    validation = PreparedDataset(recipe['data'], 'prose', 'validation', recipe['model']['context_length'])
    tokenizer_sha = sha256(recipe['tokenizer'])
    manifest_sha = sha256(Path(recipe['data']) / 'manifest.json')
    if tokenizer_sha != state['tokenizer_sha256'] or manifest_sha != state['prepared_manifest_sha256']:
        raise ValueError('Parent data/tokenizer differs from mixture inputs')
    if (Tokenizer.from_file(recipe['tokenizer']).get_vocab_size() != recipe['model']['vocab_size']
            or recipe['model'] != state['model_config']):
        raise ValueError('Model/tokenizer configuration mismatch')
    spans = read_spans(cfg['selected_spans']) if cfg['selected_fraction'] else None
    selection_record = selected_manifest(cfg, recipe, spans)
    identity = dict(pilot_config=cfg, parent=parent_record, recipe=recipe,
        tokenizer_sha256=tokenizer_sha, prepared_manifest_sha256=manifest_sha,
        selected_spans_sha256=sha256(cfg['selected_spans']) if spans is not None else None,
        selected_manifest=selection_record,
        inputs_sha256={**training.hashes, **validation.hashes}, code_sha256=code_hashes(),
        trainer_sha256=sha256(__file__), helpers_sha256={name: sha256(ROOT / 'scripts' / name)
            for name in (*targeted.DEPENDENCIES, 'train_targeted10.py', 'rebalance11_stream.py')},
        environment=environment(device, precision))
    fingerprint = targeted.digest(identity)
    if resume and state['fingerprint'] != fingerprint:
        raise ValueError('Exact mixture resume requires unchanged code/data/environment')
    stream_state = state['stream'] if resume else None
    ordinary = EpochStream(training, recipe['seed'], stream_state['ordinary'] if resume else state['stream'])
    selected = SpanStream(training, spans, cfg['selected_seed'], stream_state['selected'] if resume else None) if spans is not None else None
    stream = MixedStream(ordinary, selected, cfg['selected_fraction'], stream_state)
    local_tokens, step = (state['local_tokens_seen'], state['step']) if resume else (0, 0)
    if (stream.total_targets != parent_record['tokens'] + local_tokens
            or stream.total_targets != state['tokens_seen'] or local_tokens > cfg['max_tokens']):
        raise ValueError('Parent/pilot/mixture counters disagree')
    if selected is not None and stream.expected_extra(cfg['max_tokens']) > 2 * selected.targets_per_epoch:
        raise ValueError('Selected quota exceeds two additional pool passes; lower fraction')
    model = GPT(ModelConfig(**recipe['model'])).to(device)
    model.load_state_dict(state['model'])
    opts = optimizers(model, recipe, device)
    if set(opts) != set(state['optimizers']):
        raise ValueError('Preserve every parent optimizer state')
    for name, opt in opts.items():
        opt.load_state_dict(state['optimizers'][name])
    lineage = deepcopy(state['lineage'])
    if not resume:
        lineage['parents'].append(dict(checkpoint_sha256=parent_record['sha256'], domain='prose',
            step=parent_record['step'], optimizer='inherited complete moments and ordinary stream',
            pilot='rebalance11', schedule='inherit', lr_multiplier=cfg['lr_multiplier']))
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
            absolute_targets=stream.total_targets, **extra))

    def status(name, **extra):
        write_json(run_dir / 'status.json', dict(status=name, step=step, fingerprint=fingerprint,
            attempt_id=attempt, consumed_seconds=elapsed(), supervised_tokens_seen=stream.total_targets,
            local_tokens_seen=local_tokens, absolute_targets=stream.total_targets,
            local_target_budget=cfg['max_tokens'], mixture=stream.statistics(), **extra))

    def save(name='last.pt'):
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Refuse nonfinite mixture weights')
        save_checkpoint(run_dir / name, dict(format='poetrygpt-v1', model=model.state_dict(),
            model_config=recipe['model'], config=recipe, pilot_kind='rebalance11', pilot_config=cfg,
            parent_recipe=parent_recipe, parent_record=parent_record,
            optimizers={k: opt.state_dict() for k, opt in opts.items()},
            tokens_seen=stream.total_targets, absolute_tokens_seen=stream.total_targets,
            local_tokens_seen=local_tokens, step=step, parent_steps=parent_record['step'],
            stream=stream.state(), rng=rng_state(), lineage=lineage, fingerprint=fingerprint,
            tokenizer_sha256=tokenizer_sha, prepared_manifest_sha256=manifest_sha,
            best_nll=best_nll, consumed_seconds=elapsed(), attempt_id=attempt))

    event('resume' if resume else 'fork', inherited_optimizer=True,
          rollback_after_step=step if resume else None, mixture=stream.statistics(),
          lr_multiplier=cfg['lr_multiplier'], target_tokens=recipe['target_tokens'], batch_size=recipe['batch_size'])
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
            lr = targeted.pilot_lr(local_tokens + count, parent_record['tokens'], parent_recipe, cfg)
            for opt in opts.values():
                for group in opt.param_groups:
                    group['lr'] = lr
                opt.zero_grad(set_to_none=True)
            nll = backward_microbatches(model, batches, count, device, precision)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['grad_clip'], error_if_nonfinite=True)
            for opt in opts.values():
                opt.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            step += 1
            local_tokens += count
            peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0.
            event('train', nll=nll, main_nll=nll, lr=lr, supervised_targets=count,
                  grad_norm_before_clip=float(norm), peak_cuda_allocated_mib=peak,
                  selected_targets=stream.extra_targets, ordinary_local_targets=local_tokens - stream.extra_targets)
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
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    train(read_json(args.config), args.run_dir, args.device, args.resume, args.stop_after)


if __name__ == '__main__':
    main()
