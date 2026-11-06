"""Freeze family-disjoint generation panels, then evaluate trusted checkpoints.

Build once with --build-spec. Default inference uses only the development panel;
--confirmation explicitly opens the reserved generation panel. Full validation
NLL includes both panels' families, so confirmation is not an untouched corpus.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer
from poetrygpt.data import PreparedDataset, read_json, sha256
from poetrygpt.evaluate import evaluate
from poetrygpt.generate import encode_continuation, generate
from poetrygpt.model import GPT, ModelConfig
from poetrygpt.runtime import load_checkpoint, resolve_device, seed_everything, write_json

SPEC = ROOT / 'configs/evaluation_pretrain_series.json'
DATA = ROOT / 'data/prepared_v7/expanded'
TOKENIZER = ROOT / 'artifacts/tokenizers_v3/selected/tokenizer.json'
SEED = 20261101


def stable(value):
    return hashlib.sha256(f'{SEED}:{value}'.encode()).hexdigest()


def sources(row):
    return sorted({s.get('dataset', 'unknown') if isinstance(s, dict) else s for s in row.get('sources', [])})


def prefix_candidate(text, tokenizer, key):
    # Whole sentences, not token-decoded partial words; avoid the opening title.
    boundaries = [m.end() for m in re.finditer(r'[.!?…][»”"]?\s+(?=[А-ЯЁ«„—])', text)]
    starts = sorted((i for i in boundaries if 200 <= i < len(text) - 600), key=lambda i: stable(f'{key}:{i}'))
    for lo, hi in ((80, 100), (70, 120)):
        for start in starts[:80]:
            for end in (e for e in boundaries if start + 120 <= e <= start + 900):
                prefix = text[start:end].strip()
                count = len(tokenizer.encode(prefix, add_special_tokens=False).ids)
                if count > hi:
                    break
                if count < lo:
                    continue
                words = re.findall(r'[А-Яа-яЁё]+', prefix)
                if len(words) < 20 or len(re.findall(r'[A-Za-z0-9]', prefix)) > 3:
                    continue
                if re.search(r'http|www|примечан|редакц|\(англ|\(франц|\[|\]|\(\s*\)|(?:^|\n)\s*\d+\)', prefix, re.I):
                    continue
                if re.match(r'^[—«„"\s]*[а-яё]', prefix):
                    continue
                if sum(w.isupper() for w in words) / len(words) > .15:
                    continue
                return prefix, start, end, count
    return None


def panel_stats(cases):
    return {'cases': len(cases), 'families': len({r['family_id'] for r in cases}),
            'authors': len({r['author'] for r in cases}),
            **{field: dict(Counter(r[field] for r in cases)) for field in ('profile', 'genre', 'author')},
            'sources': dict(Counter(s for r in cases for s in r['sources'])),
            'prefix_tokens': {'min': min(r['prefix_tokens'] for r in cases), 'max': max(r['prefix_tokens'] for r in cases)}}


def build_spec():
    if SPEC.exists():
        raise ValueError('Frozen spec already exists; do not regenerate during model selection')
    dataset = PreparedDataset(DATA, 'prose', 'validation')
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    candidates, rejected = [], 0
    for meta in dataset.metadata:
        text = tokenizer.decode(dataset.ids[meta['start']:meta['end']].tolist(), skip_special_tokens=True)
        result = prefix_candidate(text, tokenizer, meta['work_id'])
        if result is None:
            rejected += 1
            continue
        prefix, start, end, count = result
        candidates.append({'id': meta['work_id'], 'family_id': meta['family_id'], 'author': meta.get('author', 'unknown'),
                           'title': meta.get('title'), 'profile': meta['profile'], 'genre': meta.get('genre', 'unknown'),
                           'sources': sources(meta), 'release_origin': meta.get('release_origin'),
                           'source_text_sha256': meta.get('text_sha256'), 'decoded_char_range': [start, end],
                           'record_token_range': [meta['start'], meta['end']],
                           'prefix': prefix, 'prefix_tokens': count, 'prefix_sha256': hashlib.sha256(prefix.encode()).hexdigest()})
    families = defaultdict(list)
    for row in candidates:
        families[row['family_id']].append(row)
    strata = defaultdict(list)
    for family, rows in families.items():
        dominant = Counter(r['profile'] for r in rows).most_common(1)[0][0]
        genre = Counter(r['genre'] for r in rows).most_common(1)[0][0]
        strata[dominant, genre].append(family)
    family_sets = [set(), set()]
    profile_family_counts = [Counter(), Counter()]
    for (profile, genre), names in sorted(strata.items()):
        for family in sorted(names, key=stable):
            side = min(range(2), key=lambda i: (profile_family_counts[i][profile], len(family_sets[i]), i))
            family_sets[side].add(family)
            profile_family_counts[side][profile] += 1
    panels = {}
    for side, name in enumerate(('development', 'confirmation')):
        selected, used = [], set()
        counts = {key: Counter() for key in ('family_id', 'author', 'genre', 'sources')}
        for profile in ('strict', 'expanded_only'):
            pool = [r for r in candidates if r['family_id'] in family_sets[side] and r['profile'] == profile]
            for _ in range(30):
                eligible = [r for r in pool if r['id'] not in used]
                if not eligible:
                    raise ValueError(f'Insufficient records for {name}/{profile}')
                def priority(row):
                    return (counts['family_id'][row['family_id']], counts['genre'][row['genre']],
                            counts['author'][row['author']], sum(counts['sources'][s] for s in row['sources']), stable(row['id']))
                row = min(eligible, key=priority)
                selected.append(row)
                used.add(row['id'])
                for field in ('family_id', 'author', 'genre'):
                    counts[field][row[field]] += 1
                counts['sources'].update(row['sources'])
        panels[name] = sorted(selected, key=lambda r: stable(r['id']))
    assert not family_sets[0] & family_sets[1]
    spec = {'version': 1, 'split': 'validation', 'test_used': False, 'seed': SEED,
            'data': DATA.relative_to(ROOT).as_posix(), 'tokenizer': TOKENIZER.relative_to(ROOT).as_posix(),
            'tokenizer_sha256': sha256(TOKENIZER), 'data_manifest_sha256': sha256(DATA / 'manifest.json'),
            'data_hashes': dataset.hashes, 'generation': {'temperature': .8, 'top_k': 40, 'max_new_tokens': 100, 'base_seeds': [7, 2026]},
            'selection': {'algorithm': 'Deterministic hashed interior sentence spans; family split stratified by dominant profile and genre; greedy diversity across families, genres, authors and sources; exactly30 cases/profile/panel.',
                          'eligible_records': len(candidates), 'rejected_records': rejected,
                          'eligible_families': len(families), 'assigned_family_counts': [len(s) for s in family_sets],
                          'limitations': ['No model output used for selection. Filters exclude obvious apparatus but are not literary proofreading.',
                                          'Only family-disjoint generation panels; authors and sources can overlap.',
                                          'Confirmation families remain in full validation NLL and may have appeared in prior project diagnostics; this is reserved generation, not an untouched test set.',
                                          'Balanced panels do not estimate corpus-frequency quality. Sentence boundaries are heuristic.']},
            'stats': {name: panel_stats(rows) for name, rows in panels.items()}, 'panels': panels}
    validate_spec(spec)
    write_json(SPEC, spec)
    print(spec['stats'])


def validate_spec(spec):
    if spec['split'] != 'validation' or spec['test_used']:
        raise ValueError('Only validation is authorized')
    if sha256(ROOT / spec['tokenizer']) != spec['tokenizer_sha256']:
        raise ValueError('Tokenizer changed')
    family_sets, record_sets, prefix_sets = [], [], []
    tokenizer = Tokenizer.from_file(str(ROOT / spec['tokenizer']))
    for rows in spec['panels'].values():
        assert len(rows) == 60 and Counter(r['profile'] for r in rows) == {'strict': 30, 'expanded_only': 30}
        assert len({r['id'] for r in rows}) == 60
        for row in rows:
            assert hashlib.sha256(row['prefix'].encode()).hexdigest() == row['prefix_sha256']
            assert len(tokenizer.encode(row['prefix'], add_special_tokens=False).ids) == row['prefix_tokens']
            assert 70 <= row['prefix_tokens'] <= 120
        family_sets.append({r['family_id'] for r in rows})
        record_sets.append({r['id'] for r in rows})
        prefix_sets.append({r['prefix_sha256'] for r in rows})
    assert not family_sets[0] & family_sets[1]
    assert not record_sets[0] & record_sets[1]
    assert not prefix_sets[0] & prefix_sets[1]


def add_profile_slices(report, metadata):
    by_id = {r['work_id']: r for r in metadata}
    for profile in ('strict', 'expanded_only'):
        rows = [r for r in report['per_record'] if by_id[r['id']]['profile'] == profile]
        if not rows:
            continue
        targets = sum(r['targets'] for r in rows)
        nll = sum(r['nll_sum'] for r in rows) / targets
        families = defaultdict(lambda: [0., 0])
        for row in rows:
            families[row['family_id']][0] += row['nll_sum']
            families[row['family_id']][1] += row['targets']
        report['metrics']['profile:' + profile] = {'nll': nll, 'perplexity': math.exp(nll), 'targets': targets,
                                                 'records': len(rows), 'families': len(families),
                                                 'family_macro_nll': sum(s / n for s, n in families.values()) / len(families)}


def repetition(text):
    words = re.findall(r'[а-яё]+', text.lower())
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return {'words': len(words), 'fourgram_repeated_share': (len(grams) - len(set(grams))) / len(grams) if grams else 0.,
            'warning': 'Lexical repetition only; zero is not evidence of semantic coherence.'}


def validate_resume(existing, identity, spec):
    """Accept only an exact completed-case prefix of this evaluation contract."""
    for field in ('training_performed', 'test_used', 'panel', 'checkpoint_sha256', 'checkpoint_step',
                  'spec_sha256', 'generation_settings', 'precision', 'evaluation_context_length', 'model_context_length'):
        if existing.get(field) != identity[field]:
            raise ValueError(f'Existing evaluation identity mismatch: {field}')
    if existing.get('status') not in ('running', 'budget_stopped', 'completed'):
        raise ValueError('Unrecognized existing evaluation status')
    rows = existing.get('samples')
    cases = spec['panels'][identity['panel']]
    seeds = spec['generation']['base_seeds']
    if not isinstance(rows, list) or len(rows) % len(seeds) or len(rows) > len(cases) * len(seeds):
        raise ValueError('Existing samples must contain complete ordered cases')
    for index, row in enumerate(rows):
        case, base_seed = cases[index // len(seeds)], seeds[index % len(seeds)]
        if any(row.get(field) != value for field, value in case.items()):
            raise ValueError(f'Existing sample provenance/order mismatch at {index}')
        expected_seed = base_seed + int(stable(case['id'])[:8], 16)
        if row.get('base_seed') != base_seed or row.get('seed') != expected_seed:
            raise ValueError(f'Existing sample seed/order mismatch at {index}')
        if not isinstance(row.get('text'), str) or type(row.get('generated_tokens')) is not int:
            raise ValueError(f'Invalid existing sample payload at {index}')
        if not 1 <= row['generated_tokens'] <= spec['generation']['max_new_tokens']:
            raise ValueError(f'Invalid existing generated token count at {index}')
    if existing.get('status') == 'completed' and len(rows) != len(cases) * len(seeds):
        raise ValueError('Completed evaluation is missing samples')
    validation = existing.get('validation')
    if validation is not None and (validation.get('coverage') != 'full' or validation.get('split') != 'validation'
                                   or validation.get('domain') != 'prose' or not validation.get('per_record')):
        raise ValueError('Existing validation is not a full prose validation result')
    if existing.get('train_probe_settings') is not None and existing['train_probe_settings'] != identity['train_probe_settings']:
        raise ValueError('Existing train probe settings or data hashes differ')
    probe = existing.get('train_probe')
    if probe is not None:
        settings = identity['train_probe_settings']
        if not settings['enabled'] or existing.get('train_probe_settings') != settings or probe.get('settings') != settings:
            raise ValueError('Existing train probe lacks matching settings/provenance')
        if probe.get('split') != 'train' or probe.get('domain') != settings['domain'] or probe.get('windows') != settings['selected_windows']:
            raise ValueError('Existing train probe report does not match its design')
        if probe.get('targets', 0) != probe.get('metrics', {}).get('overall', {}).get('targets') or probe['targets'] <= 0:
            raise ValueError('Existing train probe target coverage invalid')
    return len(rows) // len(seeds)


def run(args):
    started = time.perf_counter()
    spec = read_json(SPEC)
    validate_spec(spec)
    dataset = PreparedDataset(ROOT / spec['data'], 'prose', 'validation')
    if dataset.hashes != spec['data_hashes'] or sha256(dataset.folder / 'manifest.json') != spec['data_manifest_sha256']:
        raise ValueError('Evaluation data changed')
    state = load_checkpoint(args.checkpoint)
    if state['tokenizer_sha256'] != spec['tokenizer_sha256']:
        raise ValueError('Checkpoint tokenizer mismatch')
    device, precision = resolve_device(args.device, args.precision)
    model = GPT(ModelConfig(**state['model_config'])).to(device).eval()
    model.load_state_dict(state['model'])
    checkpoint_step = state['step']
    train_config, train_manifest_sha = state['config'], state['prepared_manifest_sha256']
    del state
    probe_batches = getattr(args, 'train_probe_batches', 256)
    if type(probe_batches) is not int or probe_batches < 0:
        raise ValueError('train_probe_batches must be a nonnegative integer; zero disables the probe')
    train_data = None
    probe_settings = {'enabled': bool(probe_batches)}
    if probe_batches:
        train_folder = ROOT / train_config['data']
        train_data = PreparedDataset(train_folder, train_config['domain'], 'train', train_config['model']['context_length'])
        if sha256(train_folder / 'manifest.json') != train_manifest_sha:
            raise ValueError('Checkpoint training data release changed')
        probe_settings.update(data=str(train_config['data']).replace('\\', '/'), domain=train_config['domain'],
                              context_length=train_data.context_length, max_batches=probe_batches, batch_size=args.batch_size,
                              selected_windows=min(len(train_data), probe_batches * args.batch_size),
                              manifest_sha256=train_manifest_sha, data_hashes=train_data.hashes,
                              selection='numpy.linspace(0, total_windows-1, selected_windows, dtype=int64)',
                              total_windows=len(train_data), precision=precision, dropout_enabled=False)
    panel = 'confirmation' if args.confirmation else 'development'
    result = {'status': 'running', 'training_performed': False, 'test_used': False, 'panel': panel,
              'checkpoint_sha256': sha256(args.checkpoint), 'checkpoint_step': checkpoint_step,
              'spec_sha256': sha256(SPEC), 'generation_settings': spec['generation'], 'precision': precision,
              'evaluation_context_length': dataset.context_length, 'model_context_length': model.config.context_length,
              'validation': None, 'samples': [], 'train_probe_settings': probe_settings, 'train_probe': None}
    completed_cases, previous_seconds = 0, 0.
    if args.output.exists():
        previous = read_json(args.output)
        completed_cases = validate_resume(previous, result, spec)
        if previous['status'] == 'completed':
            if not args.skip_validation and previous.get('validation') is None:
                raise ValueError('Completed generation-only output cannot satisfy full-validation request')
            if not probe_batches or previous.get('train_probe') is not None:
                print(f'Already completed and identities verified: {args.output}', flush=True)
                return
        result = previous
        previous_seconds = result.get('evaluation_seconds', 0.)
        result.update(status='running', resumed_completed_cases=completed_cases)
        result['train_probe_settings'] = probe_settings
        result.setdefault('train_probe', None)
        print(f'Resuming {panel} after {completed_cases} verified completed cases', flush=True)
    deadline = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00')) if args.deadline_utc else None
    if deadline is not None and deadline.tzinfo is None:
        raise ValueError('Deadline must include timezone')
    def budget_expired():
        if deadline is None or datetime.now(timezone.utc) < deadline:
            return False
        result.update(status='budget_stopped', evaluation_seconds=previous_seconds + time.perf_counter() - started,
                      deadline_utc=args.deadline_utc)
        write_json(args.output, result)
        return True
    if budget_expired():
        return
    if probe_batches and result['train_probe'] is None:
        probe = evaluate(model, train_data, device, precision, args.batch_size, max_batches=probe_batches)
        if all(row.get('profile') in ('strict', 'expanded_only') for row in train_data.metadata):
            add_profile_slices(probe, train_data.metadata)
        probe.update(settings=probe_settings, targets=probe['metrics']['overall']['targets'],
                     design='Fixed evenly spaced training windows in model.eval(), dropout disabled; no gradients/updates. '
                            'Comparable across checkpoints with identical probe settings; not a matched-family train/validation gap.')
        result['train_probe'] = probe
        result['evaluation_seconds'] = previous_seconds + time.perf_counter() - started
        write_json(args.output, result)
    if budget_expired():
        return
    if not args.skip_validation and result['validation'] is None:
        result['validation'] = evaluate(model, dataset, device, precision, args.batch_size)
        add_profile_slices(result['validation'], dataset.metadata)
        result['evaluation_seconds'] = previous_seconds + time.perf_counter() - started
        write_json(args.output, result)
    tokenizer = Tokenizer.from_file(str(ROOT / spec['tokenizer']))
    settings = spec['generation']
    stops = [tokenizer.token_to_id(t) for t in ('<|eos|>', '<|eot|>')]
    forbidden = [tokenizer.token_to_id(t) for t in ('<|pad|>', '<|bos|>', '<|user|>', '<|assistant|>')]
    for case in spec['panels'][panel][completed_cases:]:
        if budget_expired():
            return
        for base_seed in settings['base_seeds']:
            seed = base_seed + int(stable(case['id'])[:8], 16)
            seed_everything(seed)
            ids, reason = generate(model, encode_continuation(tokenizer, case['prefix']), settings['max_new_tokens'],
                                   settings['temperature'], settings['top_k'], stops, forbidden, precision)
            text = tokenizer.decode(ids, skip_special_tokens=True)
            result['samples'].append({**case, 'base_seed': base_seed, 'seed': seed, 'text': text,
                                      'generated_tokens': len(ids), 'stop_reason': reason, 'automatic': repetition(text)})
        result['evaluation_seconds'] = previous_seconds + time.perf_counter() - started
        write_json(args.output, result)
        print(f"{panel}: {len(result['samples'])}/120", flush=True)
    result.update(status='completed', evaluation_seconds=previous_seconds + time.perf_counter() - started,
                  quality_review='pending: blind human/agent review required; automatic repetition is not semantic scoring')
    write_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-spec', action='store_true')
    parser.add_argument('--validate-spec', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--confirmation', action='store_true')
    parser.add_argument('--skip-validation', action='store_true', help='Generate only; default includes full sliced validation')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--train-probe-batches', type=int, default=256,
                        help='Fixed evenly spaced train eval batches; default256 (4096 windows at batch16), zero disables')
    parser.add_argument('--deadline-utc', help='Optional timezone-aware ISO deadline; checked before validation and each case')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.build_spec:
        build_spec()
    elif args.validate_spec:
        validate_spec(read_json(SPEC))
        print('Specification checks passed')
    elif args.checkpoint and args.output:
        run(args)
    else:
        parser.error('Use --build-spec, --validate-spec or --checkpoint with --output')


if __name__ == '__main__':
    main()
