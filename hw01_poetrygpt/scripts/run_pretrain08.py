"""Sequential detached stage-0 or A/B queue; never selects a winner by loss."""
import argparse
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poetrygpt.data import read_json, sha256
from poetrygpt.runtime import write_json

AUDIT = ROOT / 'audit/pretrain08'


def command(args):
    subprocess.run([sys.executable, '-B', '-X', 'utf8', *args], cwd=ROOT, check=True)


def verify_freeze():
    frozen = read_json(AUDIT / 'freeze.json')
    for name, expected in frozen['paths_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('Frozen series input changed: ' + name)
    write_json(AUDIT / 'remote_inputs_verified.json', frozen)


def profile():
    verify_freeze()
    reports = {}
    for name in ('a_66m_adamw', 'b_68m_adamw', 'b_68m_muon'):
        cfg_path = ROOT / 'configs/training_runs/pretrain08' / (name + '.json')
        if name.endswith('muon'):
            cfg = read_json(cfg_path.with_name('b_68m_adamw.json'))
            cfg['optimizer'] = 'muon'
            cfg_path = AUDIT / 'profile_muon_config.json'
            write_json(cfg_path, cfg)
        run = ROOT / 'runs/pretrain08_profile' / name
        base = ['scripts/train_pretrain08.py', '--config', str(cfg_path), '--run-dir', str(run),
                '--until-tokens', '210000000']
        if not (run / 'last.pt').exists():
            command(base + ['--stop-after', '100'])
        command(base + ['--resume', '--stop-after', '200'])
        rows = [__import__('json').loads(line) for line in (run / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        updates = {row['step']: row for row in rows if row['event'] == 'train'}
        if len(updates) != 200 or read_json(run / 'status.json')['step'] != 200:
            raise ValueError('Incomplete GPU profile')
        measured = [updates[i] for i in range(121, 201)]
        reports[name] = dict(updates=200, resume_at_step=100,
            median_seconds=statistics.median(r['seconds'] for r in measured),
            mean_targets_per_second=sum(r['supervised_targets'] for r in measured) / sum(r['seconds'] for r in measured),
            peak_allocated_mib=max(r['peak_cuda_allocated_mib'] for r in updates.values()),
            last_nll=updates[200]['nll'], meaning='execution/resource test only; weights excluded from A/B')
        write_json(AUDIT / 'gpu_profile.json', reports)
    print(reports, flush=True)


def compare_ab():
    verify_freeze()
    if not (AUDIT / 'gpu_profile.json').exists():
        raise ValueError('Complete GPU profile before A/B')
    for name in ('a_66m_adamw', 'b_68m_adamw'):
        run = ROOT / 'runs/pretrain08' / name
        status = read_json(run / 'status.json') if (run / 'status.json').exists() else {}
        if status.get('status') != 'comparison_budget_reached' or status.get('supervised_tokens_seen') != 210000000:
            args = ['scripts/train_pretrain08.py', '--config', f'configs/training_runs/pretrain08/{name}.json',
                    '--run-dir', str(run), '--until-tokens', '210000000']
            if (run / 'last.pt').exists():
                args += ['--resume']
            write_json(AUDIT / 'queue_status.json', dict(phase='training', run=name))
            command(args)
            status = read_json(run / 'status.json')
            if status['status'] != 'comparison_budget_reached' or status['supervised_tokens_seen'] != 210000000:
                raise RuntimeError('Training stopped before comparison budget; inspect before next run')
        output = AUDIT / (name + '_210m.json')
        if not output.exists() or not read_json(output).get('complete'):
            write_json(AUDIT / 'queue_status.json', dict(phase='evaluation', run=name))
            command(['scripts/evaluate_learning_series04.py', '--checkpoint', str(run / 'last.pt'),
                     '--output', str(output), '--panel', str(AUDIT / 'eval_panel.json'),
                     '--max-wall-seconds', '1800'])
    write_json(AUDIT / 'queue_status.json', dict(phase='awaiting_quality_review',
        next='Compare A/B metrics and blind generations before choosing architecture for Muon'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['profile', 'ab'])
    args = parser.parse_args()
    profile() if args.action == 'profile' else compare_ab()


if __name__ == '__main__':
    main()
