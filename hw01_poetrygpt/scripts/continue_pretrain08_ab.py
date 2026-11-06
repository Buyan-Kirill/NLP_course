"""Continue the approved close A/B comparison to 420M, one GPU job at a time."""
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poetrygpt.data import read_json, sha256
from poetrygpt.runtime import write_json
from scripts.run_pretrain08 import command, verify_freeze

AUDIT = ROOT / 'audit/pretrain08'
NAMES = ('a_66m_adamw', 'b_68m_adamw')
BUDGET = 420000000


def preserve_210m(name):
    run = ROOT / 'runs/pretrain08' / name
    report = read_json(AUDIT / (name + '_210m.json'))
    if not report.get('complete') or report['tokens_seen'] != 210000000:
        raise ValueError('Need complete initial comparison: ' + name)
    expected = report['checkpoint_sha256']
    archive = run / 'stages/210m.pt'
    if not archive.exists():
        source = run / 'last.pt'
        if sha256(source) != expected:
            raise ValueError('Initial checkpoint no longer available: ' + name)
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix('.partial')
        if partial.exists():
            raise ValueError('Inspect interrupted checkpoint copy: ' + str(partial))
        shutil.copyfile(source, partial)
        if sha256(partial) != expected:
            raise ValueError('Checkpoint copy hash mismatch')
        partial.rename(archive)
    if sha256(archive) != expected:
        raise ValueError('Saved 210M checkpoint differs from evaluated weights')
    return dict(path=str(archive.relative_to(ROOT)), sha256=expected, bytes=archive.stat().st_size)


def evaluate(name, script, extra):
    run = ROOT / 'runs/pretrain08' / name
    suffix = '420m_quality' if 'causal' in script else '420m'
    output = AUDIT / (name + '_' + suffix + '.json')
    if output.exists():
        report = read_json(output)
        if not report.get('complete') or report['checkpoint_sha256'] != sha256(run / 'last.pt'):
            raise ValueError('Inspect partial or mismatched evaluation: ' + str(output))
        return
    write_json(AUDIT / 'continuation420_status.json', dict(phase='evaluation', run=name, output=str(output)))
    command([script, '--checkpoint', str(run / 'last.pt'), '--output', str(output), *extra])
    if not read_json(output).get('complete'):
        raise RuntimeError('Evaluation did not finish')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print('Preserve evaluated 210M checkpoints; A resume -> evaluate -> B resume -> evaluate; stop at 420M.')
        return
    decision = read_json(AUDIT / 'decision210.json')
    if decision.get('next') != 'continue_both_to_420m':
        raise ValueError('Need recorded review decision before continuation')
    if read_json(AUDIT / 'quality_queue_status.json')['phase'] != 'complete':
        raise ValueError('Initial quality queue is incomplete')
    verify_freeze()
    archives = {name: preserve_210m(name) for name in NAMES}
    write_json(AUDIT / 'checkpoints210.json', archives)
    for name in NAMES:
        run = ROOT / 'runs/pretrain08' / name
        state = read_json(run / 'status.json')
        if not 210000000 <= state['supervised_tokens_seen'] <= BUDGET:
            raise ValueError('Unexpected continuation state')
        if state['supervised_tokens_seen'] < BUDGET:
            write_json(AUDIT / 'continuation420_status.json', dict(phase='training', run=name))
            command(['scripts/train_pretrain08.py', '--config', f'configs/training_runs/pretrain08/{name}.json',
                     '--run-dir', str(run), '--until-tokens', str(BUDGET), '--resume'])
        state = read_json(run / 'status.json')
        if state['status'] != 'comparison_budget_reached' or state['supervised_tokens_seen'] != BUDGET:
            raise RuntimeError('Stopped before target; inspect before continuing queue')
        evaluate(name, 'scripts/evaluate_learning_series04.py',
                 ['--panel', str(AUDIT / 'eval_panel.json'), '--max-wall-seconds', '1800'])
        evaluate(name, 'scripts/evaluate_causal_series.py', ['--metrics-only', '--split', 'validation'])
    write_json(AUDIT / 'continuation420_status.json', dict(phase='awaiting_quality_review', targets=BUDGET))


if __name__ == '__main__':
    main()
