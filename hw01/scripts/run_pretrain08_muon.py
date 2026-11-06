"""Run the selected architecture with Muon, stopping at a reviewed token budget."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poetrygpt.data import read_json, sha256
from poetrygpt.runtime import write_json
from scripts.run_pretrain08 import command, verify_freeze
from scripts.continue_pretrain08_ab import preserve_210m

AUDIT = ROOT / 'audit/pretrain08'


def checked_evaluation(checkpoint, output, script, extra):
    expected = sha256(checkpoint)
    if not output.exists():
        command([script, '--checkpoint', str(checkpoint), '--output', str(output), *extra])
    report = read_json(output)
    if not report.get('complete') or report['checkpoint_sha256'] != expected:
        raise RuntimeError('Incomplete or mismatched evaluation; inspect before continuing: ' + str(output))


def reviewed_210m(name):
    main = read_json(AUDIT / (name + '_210m.json'))
    quality = read_json(AUDIT / (name + '_210m_quality.json'))
    if (not main.get('complete') or main.get('tokens_seen') != 210000000
            or main.get('test_used') is not False
            or not quality.get('complete') or quality.get('test_used') is not False
            or quality.get('split') != 'validation'
            or quality.get('checkpoint_sha256') != main['checkpoint_sha256']
            or quality.get('tokenizer_sha256') != main['tokenizer_sha256']):
        raise ValueError('Both 210M diagnostics must be complete and match the evaluated weights')
    # The archive remains the reference when last.pt has advanced during a resumed 420M run.
    return preserve_210m(name)


def profile(cfg_path, name):
    run = ROOT / 'runs/pretrain08_profile' / name
    output = AUDIT / (name + '_profile.json')
    if output.exists():
        if read_json(output)['config_sha256'] != sha256(cfg_path):
            raise ValueError('Profile configuration changed')
        return
    args = ['scripts/train_pretrain08.py', '--config', str(cfg_path), '--run-dir', str(run),
            '--until-tokens', '210000000', '--stop-after', '200']
    command(args + (['--resume'] if (run / 'last.pt').exists() else []))
    state = read_json(run / 'status.json')
    if state['step'] != 200:
        raise RuntimeError('Incomplete resource profile')
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    updates = {r['step']: r for r in rows if r['event'] == 'train' and r['step'] <= 200}
    if set(updates) != set(range(1, 201)):
        raise ValueError('Incomplete profile log')
    measured = [updates[i] for i in range(121, 201)]
    write_json(output, dict(config_sha256=sha256(cfg_path), measured_steps=[121, 200],
        median_seconds=statistics.median(r['seconds'] for r in measured),
        mean_targets_per_second=sum(r['supervised_targets'] for r in measured) / sum(r['seconds'] for r in measured),
        peak_allocated_mib=max(r['peak_cuda_allocated_mib'] for r in updates.values()),
        meaning='Resource test only; main run starts independently from random weights'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until-tokens', type=int, choices=(210000000, 420000000), default=210000000)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print('Requires decision420.json; validate recipe -> separate 200-step profile -> scratch Muon 210M -> metrics -> stop. 420M requires separate review decision.')
        return
    decision = read_json(AUDIT / 'decision420.json')
    base = decision['selected_architecture']
    if decision.get('next') != 'compare_muon' or base not in ('a_66m_adamw', 'b_68m_adamw'):
        raise ValueError('Need reviewed architecture choice')
    if read_json(AUDIT / 'continuation420_status.json')['phase'] != 'awaiting_quality_review':
        raise ValueError('A/B420 queue not finished')
    verify_freeze()
    for candidate in ('a_66m_adamw', 'b_68m_adamw'):
        for suffix in ('420m', '420m_quality'):
            report = read_json(AUDIT / (candidate + '_' + suffix + '.json'))
            if not report.get('complete') or report['checkpoint_sha256'] != sha256(ROOT / 'runs/pretrain08' / candidate / 'last.pt'):
                raise ValueError('Need intact evaluated A/B420 checkpoints')
    name = 'c_' + base.replace('_adamw', '_muon')
    cfg_path = ROOT / 'configs/training_runs/pretrain08' / (name + '.json')
    cfg = read_json(ROOT / 'configs/training_runs/pretrain08' / (base + '.json'))
    cfg['optimizer'] = 'muon'
    # Optional safety time cap is fixed before scratch initialization, not changed on resume.
    cfg['max_active_seconds'] = decision.get('muon_max_active_seconds', cfg['max_active_seconds'])
    if type(cfg['max_active_seconds']) is not int or not 21600 <= cfg['max_active_seconds'] <= 32400:
        raise ValueError('Unexpected resource cap')
    if cfg_path.exists():
        if read_json(cfg_path) != cfg:
            raise ValueError('Refuse to change existing Muon recipe')
    else:
        write_json(cfg_path, cfg)
    if args.until_tokens == 420000000:
        followup = read_json(AUDIT / 'decision_muon210.json')
        if followup.get('next') != 'continue_c_to_420m' or followup.get('run') != name:
            raise ValueError('Need separate 210M quality review')
        write_json(AUDIT / 'muon_checkpoint210.json', reviewed_210m(name))
    write_json(AUDIT / 'muon_queue_status.json', dict(phase='profile', run=name))
    profile(cfg_path, name)
    run = ROOT / 'runs/pretrain08' / name
    state = read_json(run / 'status.json') if (run / 'status.json').exists() else {}
    if state.get('supervised_tokens_seen', 0) > args.until_tokens:
        raise ValueError('Requested budget precedes existing run')
    if (state.get('status') != 'comparison_budget_reached'
            or state.get('supervised_tokens_seen', 0) < args.until_tokens):
        write_json(AUDIT / 'muon_queue_status.json', dict(phase='training', run=name, targets=args.until_tokens))
        command(['scripts/train_pretrain08.py', '--config', str(cfg_path), '--run-dir', str(run),
                 '--until-tokens', str(args.until_tokens)] + (['--resume'] if (run / 'last.pt').exists() else []))
    state = read_json(run / 'status.json')
    if state['status'] != 'comparison_budget_reached' or state['supervised_tokens_seen'] != args.until_tokens:
        raise RuntimeError('Training stopped early; no downstream action')
    prefix = name + '_' + str(args.until_tokens // 1000000) + 'm'
    write_json(AUDIT / 'muon_queue_status.json', dict(phase='evaluation', run=name))
    checked_evaluation(run / 'last.pt', AUDIT / (prefix + '.json'),
        'scripts/evaluate_learning_series04.py', ['--panel', str(AUDIT / 'eval_panel.json'), '--max-wall-seconds', '1800'])
    checked_evaluation(run / 'last.pt', AUDIT / (prefix + '_quality.json'),
        'scripts/evaluate_causal_series.py', ['--metrics-only', '--split', 'validation'])
    write_json(AUDIT / 'muon_queue_status.json', dict(phase='awaiting_quality_review', run=name, targets=args.until_tokens))


if __name__ == '__main__':
    main()
