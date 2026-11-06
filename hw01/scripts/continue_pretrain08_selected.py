"""Continue one reviewed pretrain08 recipe to one explicit gate, then stop.

The decision names the reviewed main report; quality/simple reports share its
stem. No model is selected here, and reaching a gate never authorizes the next.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poetrygpt.data import read_json, sha256
from poetrygpt.runtime import load_checkpoint, run_lock, write_json
from scripts.run_pretrain08 import command, verify_freeze
from scripts.run_pretrain08_muon import checked_evaluation
from scripts.train_pretrain08 import config as normalized_config

AUDIT = ROOT / 'audit/pretrain08'
NAMES = ('a_66m_adamw', 'b_68m_adamw', 'c_a_66m_muon', 'd_b_68m_muon')
THREE_EPOCH_TARGETS = 630193557
STAGES = {420000000: '420m', THREE_EPOCH_TARGETS: '630m',
          734998784: '735m', 1050000000: '1050m'}
TRANSITIONS = {(420000000, THREE_EPOCH_TARGETS),
               (420000000, 734998784), (734998784, 1050000000)}
BATCH_TARGETS = 16384
FINAL_TARGETS = 1050000000


def report_paths(main):
    return {'main': main, 'quality': main.with_name(main.stem + '_quality.json'),
            'simple': main.with_name(main.stem + '_simple.json')}


def validate_decision(decision):
    reviewed, until = decision.get('reviewed_tokens'), decision.get('until_tokens')
    if (decision.get('next') != 'continue_selected' or decision.get('run') not in NAMES
            or type(reviewed) is not int or type(until) is not int
            or (reviewed, until) not in TRANSITIONS):
        raise ValueError('Need a separate reviewed decision for exactly one allowed stage')
    digest = decision.get('checkpoint_sha256', '')
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('Decision must bind the reviewed checkpoint SHA256')
    filename = decision.get('reviewed_report')
    if not isinstance(filename, str) or not filename:
        raise ValueError('Decision must name the reviewed main report')
    main = (ROOT / filename).resolve()
    expected = AUDIT / f"{decision['run']}_{STAGES[reviewed]}.json"
    if main != expected.resolve():
        raise ValueError('Reviewed report must match run and reviewed stage in audit/pretrain08')
    # 420M already ended in a short batch. The next gate preserves full updates
    # relative to that saved cursor, not relative to a hypothetical step zero.
    # The explicitly approved three-epoch endpoint may finish in a short batch.
    # It has no outgoing transition: continuing it requires a new reviewed design.
    if until not in (FINAL_TARGETS, THREE_EPOCH_TARGETS) and (until - reviewed) % BATCH_TARGETS:
        raise ValueError('Intermediate gate must end after whole resumed batches')
    return report_paths(main)


def validate_report(path, kind, checkpoint_sha, tokenizer_sha, tokens):
    report = read_json(path)
    if (report.get('complete') is not True or report.get('training') is not False
            or report.get('test_used') is not False
            or report.get('checkpoint_sha256') != checkpoint_sha
            or report.get('tokenizer_sha256') != tokenizer_sha):
        raise ValueError('Incomplete, reserved or mismatched report: ' + str(path))
    if kind == 'quality':
        if (report.get('split') != 'validation' or report.get('metrics_only') is not True
                or report.get('quick') is not False):
            raise ValueError('Require complete validation metrics-only quality report')
        # This evaluator does not emit tokens_seen; checkpoint SHA binds its stage.
    else:
        panel = AUDIT / ('eval_panel.json' if kind == 'main' else 'simple_panel.json')
        if (report.get('tokens_seen') != tokens
                or report.get('panel_sha256') != sha256(panel)):
            raise ValueError('Wrong token budget or frozen panel: ' + str(path))
    return report


def preserve_checkpoint(source, archive, expected, dry_run=False):
    """Archive the entire resume state; retain any failed partial for inspection."""
    partial = archive.with_suffix('.partial')
    if partial.exists():
        raise FileExistsError('Inspect existing partial archive: ' + str(partial))
    if archive.exists():
        if sha256(archive) != expected:
            raise ValueError('Existing archive differs from reviewed checkpoint')
    else:
        if sha256(source) != expected:
            raise ValueError('Reviewed checkpoint unavailable; refuse to archive newer weights')
        if not dry_run:
            archive.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as src, partial.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            if sha256(partial) != expected:
                raise ValueError('Checkpoint copy SHA256 mismatch; partial retained')
            if archive.exists():
                raise FileExistsError('Archive appeared during copy; inspect both files')
            partial.rename(archive)
    return {'path': str(archive.relative_to(ROOT)), 'sha256': expected}


def checkpoint_metadata(path):
    state = load_checkpoint(path)
    return {key: state[key] for key in
            ('tokens_seen', 'config', 'model_config', 'tokenizer_sha256', 'fingerprint')}


def inspect_run(run, cfg, reviewed, until, tokenizer_sha):
    state = checkpoint_metadata(run / 'last.pt')
    status = read_json(run / 'status.json')
    tokens, reported = state['tokens_seen'], status.get('supervised_tokens_seen')
    if (type(tokens) is not int or type(reported) is not int
            or not reviewed <= tokens <= until or not reviewed <= reported <= until):
        raise ValueError('Requested budget precedes run or saved cursor is outside approved stage')
    declared_short_end = tokens == until and until in (FINAL_TARGETS, THREE_EPOCH_TARGETS)
    if not declared_short_end and (tokens - reviewed) % BATCH_TARGETS:
        raise ValueError('Saved cursor contains an undeclared short intermediate update')
    if (status.get('fingerprint') != state['fingerprint']
            or state['config'] != {**cfg, 'domain': 'prose'}
            or read_json(run / 'config.json') != cfg
            or state['model_config'] != cfg['model']
            or state['tokenizer_sha256'] != tokenizer_sha):
        raise ValueError('Resume requires the unchanged recipe, architecture and tokenizer')
    # The trainer atomically saves weights before updating status. A crash in
    # between can leave even the previous completed-stage status behind them.
    # Trust the checked checkpoint cursor; reject a completed status ahead of it.
    if status.get('status') == 'comparison_budget_reached' and reported > tokens:
        raise ValueError('Completed status disagrees with saved checkpoint')
    return tokens, status


def check_frozen_inputs(dry_run):
    if not dry_run:
        verify_freeze()
        return
    # Existing verify_freeze also writes remote_inputs_verified.json. A dry run
    # performs the same hash checks without that side effect or any lock files.
    for name, expected in read_json(AUDIT / 'freeze.json')['paths_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('Frozen series input changed: ' + name)


def continue_selected(decision_path, dry_run=False):
    queue_lock = nullcontext() if dry_run else run_lock(ROOT / 'runs/pretrain08', 'selected_queue')
    with queue_lock:
        decision = read_json(decision_path)
        reviewed_paths = validate_decision(decision)
        name, reviewed, until = (decision[k] for k in ('run', 'reviewed_tokens', 'until_tokens'))
        run = ROOT / 'runs/pretrain08' / name
        cfg_path = ROOT / 'configs/training_runs/pretrain08' / (name + '.json')
        cfg = normalized_config(read_json(cfg_path))
        if cfg['schedule_tokens'] != FINAL_TARGETS or cfg['target_tokens'] != BATCH_TARGETS:
            raise ValueError('Preserve the original 1.05B schedule and target batch')
        check_frozen_inputs(dry_run)
        tokenizer_sha = sha256(ROOT / cfg['tokenizer'])
        for kind, path in reviewed_paths.items():
            validate_report(path, kind, decision['checkpoint_sha256'], tokenizer_sha, reviewed)
        outputs = report_paths(AUDIT / f'{name}_{STAGES[until]}.json')
        # Protect inspection/copy from an existing trainer. Subprocesses acquire
        # this same series lock themselves after the archive is safe.
        guard = nullcontext() if dry_run else run_lock(ROOT / 'runs/pretrain_series01', 'series')
        with guard:
            tokens, status = inspect_run(run, cfg, reviewed, until, tokenizer_sha)
            archive = preserve_checkpoint(run / 'last.pt', run / f'stages/{STAGES[reviewed]}.pt',
                                          decision['checkpoint_sha256'], dry_run)
            current_sha = sha256(run / 'last.pt')
            if tokens == reviewed and current_sha != decision['checkpoint_sha256']:
                raise ValueError('Starting weights differ from reviewed checkpoint')
            for kind, path in outputs.items():
                if path.exists():
                    if tokens != until:
                        raise ValueError('Future-stage report exists before its checkpoint: ' + str(path))
                    validate_report(path, kind, current_sha, tokenizer_sha, until)
        plan = dict(run=name, reviewed_tokens=reviewed, until_tokens=until,
                    stage=STAGES[until], schedule_tokens=FINAL_TARGETS,
                    archive=archive, outputs={k: str(p.relative_to(ROOT)) for k, p in outputs.items()},
                    decision_sha256=sha256(decision_path), config_sha256=sha256(cfg_path))
        if dry_run:
            return dict(phase='dry_run', current_tokens=tokens, **plan)

        queue = AUDIT / 'selected_queue_status.json'
        if (tokens < until or status.get('status') != 'comparison_budget_reached'
                or status.get('supervised_tokens_seen') != tokens):
            write_json(queue, dict(phase='training', **plan))
            command(['scripts/train_pretrain08.py', '--config', str(cfg_path), '--run-dir', str(run),
                     '--until-tokens', str(until), '--resume'])
            tokens, status = inspect_run(run, cfg, reviewed, until, tokenizer_sha)
        if (tokens != until or status.get('status') != 'comparison_budget_reached'
                or status.get('supervised_tokens_seen') != tokens):
            write_json(queue, dict(phase='stopped_before_gate', current_tokens=tokens, **plan))
            raise RuntimeError('Training stopped before approved gate; no evaluation or next stage')

        checkpoint = run / 'last.pt'
        expected = sha256(checkpoint)
        jobs = [
            ('main', 'scripts/evaluate_learning_series04.py',
             ['--panel', str(AUDIT / 'eval_panel.json'), '--max-wall-seconds', '1800']),
            ('quality', 'scripts/evaluate_causal_series.py', ['--metrics-only', '--split', 'validation']),
            ('simple', 'scripts/evaluate_learning_series04.py',
             ['--panel', str(AUDIT / 'simple_panel.json'), '--skip-short-context', '--max-wall-seconds', '1800']),
        ]
        for kind, script, extra in jobs:
            if sha256(checkpoint) != expected:
                raise ValueError('Checkpoint changed during evaluation queue')
            write_json(queue, dict(phase='evaluation', diagnostic=kind, **plan))
            checked_evaluation(checkpoint, outputs[kind], script, extra)
            validate_report(outputs[kind], kind, expected, tokenizer_sha, until)
        if sha256(checkpoint) != expected:
            raise ValueError('Checkpoint changed during evaluation queue')
        result = dict(phase='awaiting_quality_review', checkpoint_sha256=expected, **plan)
        write_json(queue, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(continue_selected(args.decision, args.dry_run), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
