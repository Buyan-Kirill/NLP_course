"""Fork a poetry checkpoint without resetting optimizer, stream, RNG or token LR.

The fork has its own identity and ancestry. Token/step counters remain cumulative
within the poetry stage, so subtract branch_start_tokens for new work. The parent
is immutable; exact resume of the fork still uses the existing poetry08 trainer.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poetrygpt.data import read_json, sha256
from poetrygpt.runtime import load_checkpoint, resolve_device, run_lock, save_checkpoint, write_json
from scripts import train_poetry08 as trainer
from scripts import shared_gpu11 as shared

ALLOWED_CHANGES = {'parent', 'target_tokens', 'batch_size', 'milestones',
                   'save_interval', 'eval_interval', 'log_interval', 'max_active_seconds'}


def validate_fork(parent, cfg, parent_hash):
    trainer.validate_checkpoint(parent, cfg, parent['tokenizer_sha256'], initial=False)
    old = trainer.config(parent['config'])
    changed = {key for key in cfg if cfg[key] != old[key]}
    if changed - ALLOWED_CHANGES:
        raise ValueError('Continuation changed numerical/data recipe: ' + str(changed - ALLOWED_CHANGES))
    if cfg['mode'] != 'pure' or cfg['optimizer'] != 'muon':
        raise ValueError('This comparison requires pure poetry and Muon + AdamW')
    if set(parent.get('optimizers', {})) != {'muon', 'adamw'}:
        raise ValueError('Both saved optimizers required')
    if parent['streams']['total_targets'] != parent['tokens_seen']:
        raise ValueError('Parent stream count mismatch')
    if not 0 < parent['tokens_seen'] < cfg['schedule_tokens']:
        raise ValueError('Parent must be inside its existing token schedule')
    if parent['tokens_seen'] not in cfg['milestones'] or any(
            parent['tokens_seen'] < x and x not in cfg['milestones'] for x in old['milestones']):
        raise ValueError('Keep the fork boundary and future original milestones')
    return dict(parent_sha256=parent_hash, branch_start_tokens=parent['tokens_seen'],
                branch_start_step=parent['step'], changes={k: [old[k], cfg[k]] for k in sorted(changed)},
                optimizer_state='restored Muon and AdamW; unchanged moment hyperparameters',
                stream_state='restored cursor, pending window and epoch', rng_state='restored',
                lr_schedule='unchanged original cumulative token schedule',
                wrapper_sha256=sha256(__file__))


@contextmanager
def fork_context(record):
    original = trainer.input_identity

    def identity(*args, **kwargs):
        result = original(*args, **kwargs)
        result['poetry16_fork'] = deepcopy(record)
        return result

    trainer.input_identity = identity
    try:
        yield
    finally:
        trainer.input_identity = original


def initialize_fork(cfg, run_dir, expected_parent_sha, device_name):
    """Import full state into a new run; never edit the original checkpoint."""
    run_dir = Path(run_dir)
    with run_lock(ROOT / 'runs/pretrain_series01', 'series'), run_lock(run_dir, 'fork'):
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError('Refuse to overwrite fork: ' + str(run_dir))
        parent_path = Path(cfg['parent'])
        if sha256(parent_path) != expected_parent_sha:
            raise ValueError('Parent SHA mismatch')
        parent = load_checkpoint(parent_path)
        record = validate_fork(parent, cfg, expected_parent_sha)
        token_hash = sha256(cfg['tokenizer'])
        if token_hash != parent['tokenizer_sha256']:
            raise ValueError('Tokenizer SHA mismatch')
        training, validation = trainer.load_datasets(cfg, token_hash)
        trainer.torch.set_num_threads(cfg['cpu_threads'])
        device, precision = resolve_device(device_name, cfg['precision'])
        with fork_context(record):
            identity = trainer.input_identity(cfg, expected_parent_sha, token_hash,
                                             training, validation, device, precision)
        for key in ('input_freeze_sha256', 'prepared_manifests_sha256', 'inputs_sha256',
                    'code_sha256', 'trainer_sha256', 'dependencies_sha256', 'environment'):
            if identity[key] != parent['identity'][key]:
                raise ValueError('Saved parent/current continuation mismatch: ' + key)
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        parent['config'] = deepcopy(cfg)
        parent['identity'] = identity
        parent['fingerprint'] = fingerprint
        parent['consumed_seconds'] = 0.
        parent['attempt_id'] = 1
        parent['lineage'] = deepcopy(parent['lineage'])
        parent['lineage']['parents'].append(dict(checkpoint_sha256=expected_parent_sha,
            domain='poetry', step=parent['step'], tokens_seen=parent['tokens_seen'],
            stage='poetry16_continuation_fork', optimizer_state='restored',
            accounting='same cumulative poetry counters; no extra parent tokens added'))
        run_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(run_dir / 'last.pt', parent)
        write_json(run_dir / 'fork.json', record)
        return record


def run(cfg, run_dir, parent_sha, until_tokens, device_name='cpu', stop_after=None):
    cfg = trainer.config(cfg)
    run_dir = Path(run_dir)
    if (run_dir / 'last.pt').exists():
        # The atomic checkpoint is authoritative if a crash precedes sidecar writing.
        saved = load_checkpoint(run_dir / 'last.pt')
        record = saved['identity']['poetry16_fork']
        del saved
        if record['parent_sha256'] != parent_sha or record['wrapper_sha256'] != sha256(__file__):
            raise ValueError('Fork parent or implementation changed')
        if not (run_dir / 'fork.json').exists():
            write_json(run_dir / 'fork.json', record)
    else:
        record = initialize_fork(cfg, run_dir, parent_sha, device_name)
    with fork_context(record):
        return trainer.train(cfg, run_dir, until_tokens, device_name, resume=True, stop_after=stop_after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--until-tokens', type=int, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    args = parser.parse_args()
    if args.device == 'cuda':
        shared.AUDIT = ROOT / 'audit/poetry16/gpu_admission.jsonl'
        trainer.check_gpu_idle = lambda: shared.admit(trainer.torch, 10240, shared.Audit(
            coordinator_sha256=sha256(__file__), phase='poetry16_train'))
    run(read_json(args.config), args.run_dir, args.parent_sha256, args.until_tokens, args.device)


if __name__ == '__main__':
    main()
