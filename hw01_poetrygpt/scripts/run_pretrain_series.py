"""One reviewed training hypothesis; persistent 16-run / 8-hour series guard."""
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poetrygpt.data import read_json, sha256
from poetrygpt.runtime import run_lock, write_json

SERIES = ROOT / 'runs/pretrain_series01'
LEDGER = SERIES / 'ledger.json'


def now():
    return datetime.now(timezone.utc)


def initialize():
    if not LEDGER.exists():
        start = now()
        write_json(LEDGER, {'started_utc': start.isoformat(),
            'deadline_utc': (start + timedelta(hours=8)).isoformat(),
            'maximum_hypotheses': 16, 'time_budget_seconds': 28800,
            'scope': 'pretrain only; one hypothesis per entry; includes analysis wall time',
            'runs': [], 'status': 'active'})
    return read_json(LEDGER)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initialize', action='store_true')
    p.add_argument('--evaluate-checkpoint', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--confirmation', action='store_true')
    p.add_argument('--name')
    p.add_argument('--config', type=Path)
    p.add_argument('--parent', type=Path)
    p.add_argument('--hypothesis')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--init-weights-only', action='store_true')
    p.add_argument('--allow-dropout-change', action='store_true')
    p.add_argument('--from-scratch', action='store_true')
    args = p.parse_args()
    if args.from_scratch and (args.parent or args.init_weights_only or args.allow_dropout_change
                              or args.evaluate_checkpoint or args.initialize):
        p.error('from-scratch cannot be combined with a parent or another initialization/evaluation mode')
    if args.allow_dropout_change and (args.init_weights_only or args.evaluate_checkpoint or args.initialize):
        p.error('allow-dropout-change is only a continuation hypothesis mode')
    with run_lock(SERIES, 'series'):
        ledger = initialize()
        if args.initialize:
            print(LEDGER, flush=True)
            return
        deadline = datetime.fromisoformat(ledger['deadline_utc'])
        if now() >= deadline or ledger['status'] != 'active':
            raise ValueError('Series closed or time budget exhausted')
        if args.evaluate_checkpoint:
            if not args.output:
                p.error('evaluation output required')
            from scripts.evaluate_pretrain_series import run
            run(SimpleNamespace(checkpoint=args.evaluate_checkpoint, output=args.output,
                confirmation=args.confirmation, skip_validation=False, device='cuda',
                precision='bf16', batch_size=16, deadline_utc=deadline.isoformat()))
            return
        if not args.name or not args.config or not args.hypothesis or (not args.from_scratch and not args.parent):
            p.error('name/config/hypothesis required; parent required unless from-scratch')
        if Path(args.name).name != args.name or args.name in ('.', '..'):
            p.error('name must be a single directory component')
        cfg = read_json(args.config)
        if cfg.get('domain') != 'prose':
            raise ValueError('This research series is restricted to prose pretraining')
        stage = SERIES / args.name
        existing = [r for r in ledger['runs'] if r['name'] == args.name]
        identity = {'config_sha256': sha256(args.config),
                    'parent_sha256': None if args.from_scratch else sha256(args.parent),
                    'mode': 'from_scratch' if args.from_scratch else 'weights_only' if args.init_weights_only else
                            'continue_dropout_change' if args.allow_dropout_change else 'continue'}
        if args.resume:
            if len(existing) != 1 or any(existing[0][k] != v for k, v in identity.items()):
                raise ValueError('Resume must preserve the exact hypothesis and config')
            entry = existing[0]
            if entry['status'] == 'completed':
                raise ValueError('Completed hypothesis cannot be replayed as recovery')
            entry['recoveries'] = entry.get('recoveries', 0) + 1
        else:
            if existing or stage.exists() or len(ledger['runs']) >= ledger['maximum_hypotheses']:
                raise ValueError('Duplicate hypothesis or run cap reached')
            entry = {'number': len(ledger['runs']) + 1, 'name': args.name,
                     'hypothesis': args.hypothesis, 'started_utc': now().isoformat(),
                     'config': str(args.config), 'parent': None if args.from_scratch else str(args.parent), **identity}
            ledger['runs'].append(entry)
        entry['status'] = 'running'
        write_json(LEDGER, ledger)
        done = threading.Event()

        def stop_at_deadline():
            while not done.wait(min(5, max(.1, (deadline - now()).total_seconds()))):
                if now() >= deadline:
                    stage.mkdir(parents=True, exist_ok=True)
                    (stage / 'STOP').write_text('Series eight-hour budget reached.\n', encoding='utf-8')
                    return

        watcher = threading.Thread(target=stop_at_deadline, daemon=True)
        watcher.start()
        started = time.monotonic()
        try:
            from poetrygpt.train import train
            if args.resume:
                kwargs = {'resume': stage / 'last.pt'}
            elif args.from_scratch:
                kwargs = {}
            elif args.init_weights_only:
                kwargs = {'init_from': args.parent}
            else:
                kwargs = {'continue_from': args.parent}
            if args.allow_dropout_change and not args.resume:
                kwargs['allow_dropout_change'] = True
            train(cfg, stage, 'cuda', **kwargs)
            entry['status'] = read_json(stage / 'status.json')['status']
        except BaseException as error:
            entry.update(status='failed', error=repr(error))
            raise
        finally:
            done.set()
            entry['attempt_seconds'] = entry.get('attempt_seconds', 0) + time.monotonic() - started
            entry['last_finished_utc'] = now().isoformat()
            if now() >= deadline:
                ledger['status'] = 'budget_exhausted'
            write_json(LEDGER, ledger)
        print('Finished one hypothesis; analysis required before another launch.', flush=True)


if __name__ == '__main__':
    main()
