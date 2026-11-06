"""Bounded, read-only learning diagnostics on a frozen train/validation panel.

Teacher forcing scores the exact supplied target tokens without adding EOS.
All context interventions share those targets; generation is a separate test.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer

from poetrygpt.data import read_json, sha256
from poetrygpt.generate import encode_continuation, generate
from poetrygpt.model import GPT, ModelConfig, token_losses
from poetrygpt.runtime import (autocast, environment, load_checkpoint, run_lock,
                               seed_everything, write_json)
from poetrygpt.serialization import special_ids
from scripts.evaluate_causal_series import (POETRY, PROSE, TOKENIZER,
                                            aggregate_likelihood, first_two_sentences,
                                            lexical_metrics, sampled_nll)
from scripts.run_task_pilot02 import require_resources
from scripts.train_causal_series import check_gpu_idle

BATCH_SIZE = 16
CONTEXT_FIELDS = {'true': 'prompt_ids', 'wrong': 'wrong_prompt_ids',
                  'reversed': 'reversed_prompt_ids'}


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_panel(panel, vocab_size):
    """Reject ambiguous or reserved inputs before any checkpoint is loaded."""
    layout = panel.get('scoring_layout', 'fixed')
    if layout not in ('fixed', 'variable'):
        raise ValueError('scoring_layout must be fixed or variable')
    cases, generation = panel['cases'], panel['generation']
    if not 0 <= len(cases) <= 4096 or not 1 <= len(generation) <= 512:
        raise ValueError('Expected 0..4096 scoring cases and 1..512 generation cases')
    for rows in (cases, generation):
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate panel IDs')
        if any(r['split'] not in ('train', 'validation') for r in rows):
            raise ValueError('Only train/validation panels are allowed; reserve is excluded')
        if any(not r.get('family_id') or not r.get('genre') for r in rows):
            raise ValueError('Each case needs a family_id and genre')
    for row in cases:
        for field, length in [('target_ids', 64)] + [(f, 256) for f in CONTEXT_FIELDS.values()]:
            ids = row[field]
            maximum = 96 if field == 'target_ids' else 416
            if (not isinstance(ids, list)
                    or not (1 <= len(ids) <= maximum if layout == 'variable' else len(ids) == length)
                    or any(type(t) is not int or not 0 <= t < vocab_size for t in ids)):
                raise ValueError(f'Invalid {field}: {row["id"]}')
        if any(len(row[field]) + len(row['target_ids']) > 512 for field in CONTEXT_FIELDS.values()):
            raise ValueError('Scoring sequence exceeds 512 tokens: ' + row['id'])
        if row['wrong_prompt_ids'] == row['prompt_ids']:
            raise ValueError('Wrong context equals true context: ' + row['id'])
        if not isinstance(row.get('exposure'), dict):
            raise ValueError('Expected exposure mapping: ' + row['id'])
    for row in generation:
        if not isinstance(row['prefix'], str) or not row['prefix'].strip():
            raise ValueError('Empty generation prefix')
        if type(row['seed']) is not int or not 0 <= row['seed'] < 2**32:
            raise ValueError('Generation seed must fit NumPy uint32')
        if 'prompt_ids' in row:
            ids = row['prompt_ids']
            if (not isinstance(ids, list) or not 1 <= len(ids) <= 416
                    or any(type(t) is not int or not 0 <= t < vocab_size for t in ids)):
                raise ValueError('Invalid exact generation prompt IDs')


def verify_sources(source_hashes):
    if not isinstance(source_hashes, dict) or not 1 <= len(source_hashes) <= 256:
        raise ValueError('Expected a bounded path-to-SHA256 source_hashes mapping')
    verified = {}
    for filename, expected in source_hashes.items():
        path = (ROOT / filename).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Panel source must be within project: ' + filename)
        if re.search(r'(^|[/\\_.-])(test|reserve)([/\\_.-]|$)', filename, re.I):
            raise ValueError('Reserved source is forbidden: ' + filename)
        digest = sha256(path)
        if digest != expected:
            raise ValueError('Panel source changed: ' + filename)
        verified[filename] = digest
    return verified


@torch.no_grad()
def score_token_targets(model, pairs, pad_id, device, precision, batch_size=BATCH_SIZE):
    """Return aligned per-token NLL/top-1; same mask/shift as score_candidates."""
    if batch_size < 1:
        raise ValueError('Positive batch size required')
    model.eval()
    results = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        if any(not p or not t or len(p) + len(t) > model.config.context_length for p, t in batch):
            raise ValueError('Empty or over-context scoring pair; no truncation')
        width = max(len(p) + len(t) for p, t in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
        labels = torch.full_like(ids, -100)
        for row, (prompt, targets) in enumerate(batch):
            ids[row, :len(prompt) + len(targets)] = torch.tensor(prompt + targets, device=device)
            labels[row, len(prompt):len(prompt) + len(targets)] = torch.tensor(targets, device=device)
        with autocast(device, precision):
            logits = model(ids)
            losses, mask = token_losses(logits, labels)
        if not torch.isfinite(losses[mask]).all():
            raise FloatingPointError('Nonfinite target NLL')
        predictions = logits[:, :-1].argmax(-1)
        correct = predictions.eq(labels[:, 1:])
        for row in range(len(batch)):
            nll = losses[row][mask[row]].float().cpu().tolist()
            hits = correct[row][mask[row]].cpu().tolist()
            results.append(dict(**aggregate_likelihood(sum(nll), len(nll)),
                                correct_tokens=sum(hits), token_accuracy=sum(hits) / len(hits),
                                token_nll=nll, token_correct=hits,
                                predicted_ids=predictions[row][mask[row]].cpu().tolist()))
    return results


def summarize_cases(rows):
    """Token-weighted likelihood; paired differences remain within each case."""
    groups = defaultdict(list)
    for row in rows:
        keys = ['overall', 'split:' + row['split'], 'genre:' + row['genre'],
                'split:' + row['split'] + '/genre:' + row['genre']]
        for checkpoint, exposure in sorted(row['exposure'].items()):
            label = json.dumps(exposure, sort_keys=True, separators=(',', ':'))
            keys.extend(['exposure:' + checkpoint + ':' + label,
                         'split:' + row['split'] + '/exposure:' + checkpoint + ':' + label])
        for key in keys:
            groups[key].append(row)
    summary = {}
    for key, selected in groups.items():
        entry = dict(cases=len(selected), families=len({r['family_id'] for r in selected}),
                     contexts={}, quarters={}, context_benefit={})
        for name in selected[0]['scores']:
            scores = [r['scores'][name] for r in selected]
            targets = sum(s['targets'] for s in scores)
            entry['contexts'][name] = dict(
                **aggregate_likelihood(sum(s['sum_nll'] for s in scores), targets),
                token_accuracy=sum(s['correct_tokens'] for s in scores) / targets)
        maximum_targets = max(len(score['token_nll']) for row in selected for score in row['scores'].values())
        for start in range(0, maximum_targets, 16):
            end = start + 16
            values = {}
            for name in selected[0]['scores']:
                nll = sum(sum(r['scores'][name]['token_nll'][start:end]) for r in selected)
                hits = sum(sum(r['scores'][name]['token_correct'][start:end]) for r in selected)
                targets = sum(len(r['scores'][name]['token_nll'][start:end]) for r in selected)
                if targets:
                    values[name] = dict(**aggregate_likelihood(nll, targets), token_accuracy=hits / targets)
            if values:
                entry['quarters'][f'{start + 1}-{end}'] = values
        for name in selected[0]['scores']:
            if name == 'true':
                continue
            differences = [r['scores'][name]['nll'] - r['scores']['true']['nll'] for r in selected]
            entry['context_benefit'][name + '_minus_true_nll'] = dict(
                mean=sum(differences) / len(differences),
                true_lower_nll_fraction=sum(d > 0 for d in differences) / len(differences),
                ties=sum(d == 0 for d in differences), paired_cases=len(differences))
        summary[key] = entry
    return summary


def reference_copy(text, case, tokenizer):
    reference = case.get('reference_text', case.get('reference'))
    if reference is None and 'reference_ids' in case:
        reference = tokenizer.decode(case['reference_ids'], skip_special_tokens=True)
    if not isinstance(reference, str):
        return dict(available=False)
    return dict(available=True, reference_text=reference,
                reference_copy4=lexical_metrics(text, reference)['prompt_copy4'],
                scope='Overlap with this case reference only; not full-corpus retrieval')


def evaluate(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError('Preserve frozen evaluation: ' + str(args.output))
    torch.set_num_threads(4)
    panel = read_json(args.panel)
    tokenizer_hash = sha256(TOKENIZER)
    if panel['tokenizer_sha256'] != tokenizer_hash:
        raise ValueError('Panel tokenizer differs from selected tokenizer')
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    validate_panel(panel, tokenizer.get_vocab_size())
    input_hashes = verify_sources(panel['source_hashes'])
    corpus_manifests = {}
    for domain, folder in [('prose', PROSE), ('poetry', POETRY)]:
        manifest = read_json(folder / 'manifest.json')
        if manifest['inputs_sha256']['artifacts/tokenizers_v3/selected/tokenizer.json'] != tokenizer_hash:
            raise ValueError('Validation corpus tokenizer mismatch: ' + domain)
        corpus_manifests[domain] = sha256(folder / 'manifest.json')
    check_gpu_idle()
    device, precision, resources = require_resources(False)
    state = load_checkpoint(args.checkpoint)
    if state['tokenizer_sha256'] != tokenizer_hash:
        raise ValueError('Checkpoint tokenizer differs from selected tokenizer')
    cfg = ModelConfig(**state['model_config'])
    required_context = max((len(row[field]) + len(row['target_ids']) for row in panel['cases']
                            for field in CONTEXT_FIELDS.values()), default=0)
    if cfg.vocab_size != tokenizer.get_vocab_size() or cfg.context_length < required_context:
        raise ValueError('Model vocabulary/context cannot score this panel')
    for case in panel['generation']:
        if len(encode_continuation(tokenizer, case['prefix'])) + 96 > cfg.context_length:
            raise ValueError('Generation budget does not fit: ' + case['id'])
    model = GPT(cfg).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    report = dict(schema=1, complete=False, training=False, test_used=False,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                  tokenizer_sha256=tokenizer_hash, panel=str(args.panel), panel_sha256=sha256(args.panel),
                  source_hashes=input_hashes, model_config=model.config_dict(),
                  corpus_manifest_sha256=corpus_manifests,
                  config=state.get('config'), lineage=state['lineage'],
                  step=state.get('step'), parent_steps=state.get('parent_steps'),
                  tokens_seen=state.get('tokens_seen'), parent_tokens_seen=state.get('parent_tokens_seen'),
                  checkpoint_fingerprint=state.get('fingerprint'),
                  config_sha256=json_sha256(state.get('config')),
                  lineage_sha256=json_sha256(state['lineage']),
                  parameters=model.parameter_count(), resources=resources,
                  environment=environment(device, precision),
                  code_sha256={name: sha256(ROOT / name) for name in (
                      'scripts/evaluate_learning_series04.py', 'scripts/evaluate_causal_series.py',
                      'poetrygpt/model.py', 'poetrygpt/generate.py', 'poetrygpt/runtime.py',
                      'poetrygpt/data.py', 'scripts/run_task_pilot02.py', 'scripts/train_causal_series.py')},
                  settings=dict(batch_size=BATCH_SIZE, scoring_layout=panel.get('scoring_layout', 'fixed'),
                                target_tokens=None if panel.get('scoring_layout') == 'variable' else 64,
                                context_tokens=None if panel.get('scoring_layout') == 'variable' else 256,
                                position_slices='16-token bins through the longest target; denominators count actual available targets',
                                short_context_tokens=None if args.skip_short_context else 32,
                                temperature=.7, top_k=40, top_p=.9, max_new_tokens=96,
                                validation_windows_per_domain=32, max_wall_seconds=args.max_wall_seconds),
                  cases=[], samples=[], likelihood={}, timings={},
                  grammar_probe=dict(performed=False, reason='Deferred; prior frozen diagnostic available'),
                  limitations=[
                      'Teacher forcing reveals true earlier target tokens; it does not measure free-generation coherence.',
                      'Positive context benefit means intervention NLL minus true-context NLL is positive.',
                      'Wrong-work and reversed-context interventions can change style and distribution as well as meaning.',
                      'Short context uses up to the final 32 tokens; absolute positions restart at zero.',
                      'Exposure strata are descriptive, not randomized causal effects.',
                      'Cases within a family are dependent; no independent-case significance claim.',
                      'Reference overlap is descriptive and is not full-corpus copy retrieval.'])
    del state
    special = special_ids(tokenizer)

    def save(stage):
        report['stage'] = stage
        report['wall_seconds'] = time.perf_counter() - started
        write_json(args.output, report)
        print(stage, round(report['wall_seconds'], 1), flush=True)

    def check_deadline():
        if time.perf_counter() - started >= args.max_wall_seconds:
            raise TimeoutError('Evaluation wall budget exhausted; partial report preserved')

    save('loaded')
    try:
        stage_start = time.perf_counter()
        for start in range(0, len(panel['cases']), BATCH_SIZE):
            check_deadline()
            batch = panel['cases'][start:start + BATCH_SIZE]
            scored = {}
            for name, field in CONTEXT_FIELDS.items():
                scored[name] = score_token_targets(model, [(r[field], r['target_ids']) for r in batch],
                                                   special['<|pad|>'], device, precision)
            if not args.skip_short_context:
                scored['last32'] = score_token_targets(
                    model, [(r['prompt_ids'][-32:], r['target_ids']) for r in batch],
                    special['<|pad|>'], device, precision)
            for i, row in enumerate(batch):
                scores = {name: values[i] for name, values in scored.items()}
                report['cases'].append(dict(**row, scores=scores, context_benefit={
                    name + '_minus_true_nll': score['nll'] - scores['true']['nll']
                    for name, score in scores.items() if name != 'true'}))
            save('teacher_forcing_' + str(len(report['cases'])))
        report['summary'] = summarize_cases(report['cases'])
        report['timings']['teacher_forcing_seconds'] = time.perf_counter() - stage_start
        save('teacher_forcing_complete')
        for domain, folder in [('prose', PROSE), ('poetry', POETRY)]:
            check_deadline()
            stage_start = time.perf_counter()
            report['likelihood'][domain + '_validation'] = sampled_nll(
                model, folder, domain, 'validation', 32, device, precision)
            report['timings'][domain + '_validation_seconds'] = time.perf_counter() - stage_start
            save(domain + '_validation')
        stage_start = time.perf_counter()
        for case in panel['generation']:
            check_deadline()
            seed_everything(case['seed'])
            prompt = case.get('prompt_ids')
            if prompt is None:
                prompt = encode_continuation(tokenizer, case['prefix'])
            ids, reason = generate(model, prompt, 96, .7, 40,
                                   [special['<|eos|>'], special['<|eot|>']],
                                   [special[t] for t in ('<|pad|>', '<|bos|>', '<|user|>', '<|assistant|>')],
                                   precision, top_p=.9)
            text = tokenizer.decode(ids, skip_special_tokens=True)
            report['samples'].append(dict(**case, text=text, first_two=first_two_sentences(text),
                                          raw_text=tokenizer.decode(ids, skip_special_tokens=False),
                                          generated_ids=ids, generated_tokens=len(ids), stop_reason=reason,
                                          **lexical_metrics(text, case['prefix']),
                                          reference_copy=reference_copy(text, case, tokenizer)))
            save('generation_' + str(len(report['samples'])))
        report['timings']['generation_seconds'] = time.perf_counter() - stage_start
        report['generation_summary'] = dict(
            cases=len(report['samples']), families=len({r['family_id'] for r in report['samples']}),
            stop_reasons=dict(Counter(r['stop_reason'] for r in report['samples'])),
            means={key: sum(values) / len(values) if values else None
                   for key in ('generated_tokens', 'words', 'distinct2', 'repeated4', 'prompt_copy4')
                   for values in [[r[key] for r in report['samples'] if r[key] is not None]]})
        report['complete'] = True
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 1024**3
        save('complete')
    except BaseException as error:
        report['error'] = repr(error)
        save('failed_or_interrupted')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--panel', type=Path, default=ROOT / 'audit/learning_series04/panel.json')
    parser.add_argument('--skip-short-context', action='store_true')
    parser.add_argument('--max-wall-seconds', type=float, default=1800)
    args = parser.parse_args()
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        parser.error('--max-wall-seconds must be positive and finite')
    with run_lock(ROOT / 'runs/pretrain_series01', 'series'):
        evaluate(args)


if __name__ == '__main__':
    main()
