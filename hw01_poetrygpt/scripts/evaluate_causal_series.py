"""Frozen, read-only quality panel for causal pretraining experiments.

Run under the shared GPU series lock. Reserve is an explicit final-only choice.
Scores are diagnostics, not automatic claims of coherent free generation.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tokenizers import Tokenizer
from poetrygpt.data import PreparedDataset, read_json, sha256
from poetrygpt.generate import encode_continuation, generate
from poetrygpt.model import GPT, ModelConfig, token_losses
from poetrygpt.runtime import autocast, load_checkpoint, run_lock, seed_everything, write_json
from poetrygpt.serialization import special_ids
from scripts.run_task_pilot02 import require_resources
from scripts.score_rublimp_probe import score_ids

TOKENIZER = ROOT / 'artifacts/tokenizers_v3/selected/tokenizer.json'
SCENES = ROOT / 'data/causal_scenes01'
PROSE = ROOT / 'data/prepared_v7/expanded'
POETRY = ROOT / 'data/prepared_v7/poetry_supplement'


def evenly_spaced(size, count):
    if size < 1 or count < 1:
        raise ValueError('Positive sample size required')
    return np.linspace(0, size - 1, min(count, size), dtype=np.int64).tolist()


def candidate_tokens(tokenizer, prefix, answer, context_length):
    prompt = encode_continuation(tokenizer, prefix)
    full = encode_continuation(tokenizer, prefix + answer)
    if full[:len(prompt)] != prompt:
        raise ValueError('BPE boundary changed: scoring must not include part of the prefix')
    if len(full) <= len(prompt) or len(full) > context_length:
        raise ValueError('Empty candidate or over context; no truncation')
    return prompt, full[len(prompt):]


@torch.no_grad()
def score_candidates(model, pairs, pad_id, device, precision, batch_size=16):
    """Exactly one causal shift; only continuation tokens contribute."""
    model.eval()
    results = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        if any(not p or not t or len(p) + len(t) > model.config.context_length for p, t in batch):
            raise ValueError('Invalid candidate sequence')
        width = max(len(p) + len(t) for p, t in batch)
        ids = torch.full((len(batch), width), pad_id, device=device, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        for row, (prompt, targets) in enumerate(batch):
            ids[row, :len(prompt) + len(targets)] = torch.tensor(prompt + targets, device=device)
            labels[row, len(prompt):len(prompt) + len(targets)] = torch.tensor(targets, device=device)
        with autocast(device, precision):
            losses, mask = token_losses(model(ids), labels)
        if not torch.isfinite(losses[mask]).all():
            raise FloatingPointError('Nonfinite candidate score')
        for row, (_, targets) in enumerate(batch):
            total = -float(losses[row].sum())
            results.append(dict(sum_logp=total, mean_logp=total / len(targets), tokens=len(targets)))
    return results


def aggregate_likelihood(total_nll, targets):
    if targets <= 0:
        raise ValueError('No scored targets')
    nll = total_nll / targets
    return dict(targets=int(targets), sum_nll=float(total_nll), nll=float(nll), perplexity=math.exp(nll))


@torch.no_grad()
def sampled_nll(model, folder, domain, split, count, device, precision):
    dataset = PreparedDataset(folder, domain, split, model.config.context_length)
    indices = evenly_spaced(len(dataset), count)
    totals = defaultdict(lambda: [0., 0])
    window_scores = []
    for start in range(0, len(indices), 8):
        chosen = indices[start:start + 8]
        ids, labels = dataset.batch(chosen, device)
        with autocast(device, precision):
            losses, masks = token_losses(model(ids), labels)
        losses, masks = losses.cpu().numpy(), masks.cpu().numpy()
        for row, index in enumerate(chosen):
            nlls = losses[row][masks[row]]
            if not np.isfinite(nlls).all():
                raise FloatingPointError('Nonfinite corpus NLL')
            window_scores.append(dict(index=index, **aggregate_likelihood(nlls.sum(), len(nlls))))
            owners, valid = dataset.target_owners(index)
            for owner in np.unique(owners[valid]):
                selected = (owners == owner) & valid
                amount, number = float(nlls[selected].sum()), int(selected.sum())
                meta = dataset.metadata[int(owner)]
                keys = ['overall', 'genre:' + str(meta.get('genre', 'unknown')),
                        'author:' + str(meta.get('author', 'unknown'))]
                keys += ['source:' + str(s.get('dataset', 'unknown')) for s in meta.get('sources', [])]
                for key in set(keys):
                    totals[key][0] += amount
                    totals[key][1] += number
    return dict(indices=indices, input_sha256=dataset.hashes,
                slices={k: aggregate_likelihood(*v) for k, v in totals.items()}, windows=window_scores,
                sampling='Fixed evenly spaced stream windows; source strata can overlap')


def candidate_result(row, scores, null_scores=None):
    result = {key: row.get(key) for key in ('id', 'group', 'skill', 'complexity')}
    result['scores'] = scores
    for metric in ('sum', 'mean'):
        values = [s[metric + '_logp'] for s in scores]
        result['correct_' + metric] = values[0] > max(values[1:])
        result['pairwise_' + metric] = sum(values[0] > x for x in values[1:]) / (len(values) - 1)
        result['margin_' + metric] = values[0] - max(values[1:])
        result['tie_' + metric] = values[0] == max(values[1:])
        if null_scores is not None:
            baseline = [s[metric + '_logp'] for s in null_scores]
            result['context_gain_' + metric] = result['margin_' + metric] - (baseline[0] - max(baseline[1:]))
    result['positive_shortest'] = scores[0]['tokens'] == min(s['tokens'] for s in scores)
    result['equal_lengths'] = len({s['tokens'] for s in scores}) == 1
    if null_scores is not None:
        result['null_scores'] = null_scores
    return result


def summarize_candidates(results):
    groups = defaultdict(list)
    for result in results:
        for key in ('overall', 'skill:' + str(result['skill']), 'complexity:' + str(result['complexity'])):
            groups[key].append(result)
    output = {}
    for key, rows in groups.items():
        values = dict(n=len(rows), groups=len({r['group'] for r in rows}),
                      equal_length_cases=sum(r['equal_lengths'] for r in rows),
                      positive_shortest_cases=sum(r['positive_shortest'] for r in rows))
        for metric in ('sum', 'mean'):
            for field in ('correct', 'pairwise', 'margin', 'tie'):
                values[field + '_' + metric] = sum(r[field + '_' + metric] for r in rows) / len(rows)
            measured = [r['context_gain_' + metric] for r in rows if 'context_gain_' + metric in r]
            values['context_gain_' + metric] = sum(measured) / len(measured) if measured else None
            values['context_gain_n'] = len(measured)
        output[key] = values
    return output


def lexical_metrics(text, prefix):
    words = re.findall(r'[\w]+', text.lower(), flags=re.UNICODE)
    prefix_words = re.findall(r'[\w]+', prefix.lower(), flags=re.UNICODE)
    def grams(items, n):
        return [tuple(items[i:i+n]) for i in range(len(items)-n+1)]
    four, two = grams(words, 4), grams(words, 2)
    prompt_four = set(grams(prefix_words, 4))
    return dict(words=len(words), distinct2=len(set(two))/len(two) if two else None,
                repeated4=1-len(set(four))/len(four) if four else 0.,
                prompt_copy4=sum(g in prompt_four for g in four)/len(four) if four else 0.)


def first_two_sentences(text):
    # Display aid only: retain raw text for judgment of truncation/grammar.
    ends = list(re.finditer(r'[.!?…]+[»”\"]?(?=\s|$)', text))
    return text[:ends[1].end()] if len(ends) >= 2 else text


def sampled_training_ngrams(count=2048, width=12):
    """Bounded copy screen, deliberately not presented as full-corpus retrieval."""
    stream = np.load(PROSE/'prose_train.npy', mmap_mode='r', allow_pickle=False)
    windows = evenly_spaced((len(stream)-1)//511, count)
    grams = set()
    for index in windows:
        tokens = stream[index*511:index*511+512]
        grams.update(tokens[i:i+width].tobytes() for i in range(len(tokens)-width+1))
    return grams, dict(windows=len(windows), sampled_positions=len(windows)*512,
                       corpus_positions=len(stream), ngram_tokens=width,
                       scope='Fixed train windows only; absence does not establish absence of copying')


def training_copy_fraction(ids, train_grams, width=12):
    tokens = np.asarray(ids, dtype=np.uint16)
    count = max(0, len(tokens)-width+1)
    matches = sum(tokens[i:i+width].tobytes() in train_grams for i in range(count))
    return dict(sampled_train_copy12=matches/count if count else 0.,
                sampled_train_matching12=matches, generated_ngrams12=count)


def reserve_natural_prefixes(tokenizer):
    """Final-only raw prefixes from 60 distinct held-out literary families."""
    stream = np.load(PROSE/'prose_test.npy', mmap_mode='r', allow_pickle=False)
    metadata = read_json(PROSE/'prose_test_offsets.json')
    manifest = read_json(PROSE/'manifest.json')
    for filename in ('prose_test.npy', 'prose_test_offsets.json'):
        if sha256(PROSE/filename) != manifest['output_sha256'][filename]:
            raise ValueError('Reserve corpus hash changed')
    chosen, families = [], set()
    # Deterministic selection independent of model output and record length.
    for row in sorted(metadata, key=lambda r: hashlib.sha256(('causal01:' + r['work_id']).encode()).hexdigest()):
        family = row.get('family_id', row['work_id'])
        if family in families:
            continue
        text = tokenizer.decode(stream[row['start']:min(row['end']-1, row['start']+192)].tolist(), skip_special_tokens=True)
        # Only complete sentence-ending segments; do not expose their continuation.
        ends = list(re.finditer(r'[.!?…]+[»”\"]?(?=\s|$)', text))
        if len(ends) < 2:
            continue
        prefix = text[:ends[1].end()]
        if len(re.findall(r'\w+', prefix)) < 12:
            continue
        families.add(family)
        chosen.append(dict(id='reserve_' + row['work_id'], prefix=prefix, section='historical_reserve', group=family,
                           author=row.get('author'), genre=row.get('genre'), sources=row.get('sources', [])))
        if len(chosen) == 60:
            return chosen
    raise ValueError('Not enough independent reserve literary prefixes')


def build_generation_panel(rows, split, tokenizer, context_length, quick):
    panel, seen = [], set()
    def add(identifier, prefix, section, group):
        if prefix in seen:
            return
        if len(encode_continuation(tokenizer, prefix)) + 96 > context_length:
            raise ValueError('Full generation budget does not fit: ' + identifier)
        seen.add(prefix)
        panel.append(dict(id=identifier, prefix=prefix, section=section, group=group))
    by_skill = defaultdict(list)
    for row in rows:
        by_skill[row['skill']].append(row)
    # Round robin prevents domination by the first skill in the source file.
    per_skill = [sorted(v, key=lambda r: r['id']) for _, v in sorted(by_skill.items())]
    ordered = [v[i] for i in range(max(map(len, per_skill))) for v in per_skill if i < len(v)]
    for row in ordered[:40]:
        add('causal_' + row['id'], row['prefix'], 'causal:' + row['skill'], row['group'])
    if split == 'reserve':
        for case in reserve_natural_prefixes(tokenizer):
            add(case['id'], case['prefix'], case['section'], case['group'])
    else:
        coherence = read_json(ROOT/'audit/coherence_diagnostic01/panel.json')
        for case in coherence['cases']:
            if case['kind'] in ('unassisted', 'new_retelling', 'familiar'):
                add(case['id'], case['prefix'], 'diagnostic:' + case['kind'], case['family'])
        for filename in ('audit/synthetic_prose_trial01/panel.json', 'audit/final_base/panels/development.json'):
            for case in read_json(ROOT/filename)['cases']:
                if len(panel) >= 100:
                    break
                add(case['id'], case['prompt'], case.get('slice', 'historical'),
                    case.get('group_id', case.get('family_id', case['id'])))
    if len(panel) < 80:
        raise ValueError(f'Insufficient distinct generation cases: {len(panel)}')
    return [panel[i] for i in evenly_spaced(len(panel), 10)] if quick else panel


def evaluate(args):
    metrics_only = getattr(args, 'metrics_only', False)
    if metrics_only and args.quick:
        raise ValueError('--metrics-only requires the full diagnostic; cannot combine with --quick')
    started = time.perf_counter()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError('Preserve frozen evaluation: ' + str(args.output))
    device, precision, resources = require_resources(False)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    state = load_checkpoint(args.checkpoint)
    if state['tokenizer_sha256'] != sha256(TOKENIZER):
        raise ValueError('Checkpoint tokenizer mismatch')
    model = GPT(ModelConfig(**state['model_config'])).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    step = state.get('step')
    del state
    special = special_ids(tokenizer)
    scene_path = SCENES / (args.split + '.jsonl')
    rows = [json.loads(s) for s in scene_path.read_text(encoding='utf-8').splitlines() if s.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate scene IDs')
    panel = [] if metrics_only else build_generation_panel(rows, args.split, tokenizer, model.config.context_length, args.quick)
    report = dict(schema=1, complete=False, training=False, split=args.split, test_used=args.split == 'reserve', quick=args.quick,
                  checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint), step=step,
                  tokenizer_sha256=sha256(TOKENIZER), script_sha256=sha256(Path(__file__)),
                  scene_sha256=sha256(scene_path), resources=resources, precision=precision,
                  generation_panel=panel, samples=[], timings={},
                  limitations=['RuBLiMP is a frozen 320-pair diagnostic, not the full benchmark.',
                    'Candidate choice and lexical measures do not establish free-generation coherence.',
                    'Historical/familiar development prompts are reused; reserve uses 40 causal scenes and 60 held-out literary families.',
                    'Sampled stream-window NLL is token-weighted; small source slices are descriptive.',
                    'Train-validation NLL gap is descriptive: genre composition is not matched.',
                    'No automatic inference of logical correctness from word overlap.'])
    if metrics_only:
        report['metrics_only'] = True
    def save(stage):
        report['stage'] = stage
        report['wall_seconds'] = time.perf_counter() - started
        write_json(args.output, report)
        print(stage, round(report['wall_seconds'], 1), flush=True)
    save('loaded')
    old_split = 'test' if args.split == 'reserve' else 'validation'
    report['likelihood'] = {}
    for name, folder, domain, split, count in (
        ('prose_train', PROSE, 'prose', 'train', 128),
        ('prose_' + old_split, PROSE, 'prose', old_split, 128),
        ('poetry_' + old_split, POETRY, 'poetry', old_split, 64)):
        report['likelihood'][name] = sampled_nll(model, folder, domain, split, 4 if args.quick else count, device, precision)
        save(name)
    report['train_validation_gap'] = (report['likelihood']['prose_' + old_split]['slices']['overall']['nll']
                                     - report['likelihood']['prose_train']['slices']['overall']['nll'])
    probe = ROOT/'audit/rublimp_probe/pairs.jsonl'
    probe_manifest = read_json(probe.parent/'manifest.json')
    if sha256(probe) != probe_manifest['panel_sha256'] or sha256(TOKENIZER) != probe_manifest['tokenizer_sha256']:
        raise ValueError('Frozen RuBLiMP inputs changed')
    probes = [json.loads(s) for s in probe.read_text(encoding='utf-8').splitlines()]
    if args.quick:
        probes = [probes[i] for i in evenly_spaced(len(probes), 8)]
    scores = score_ids(model, [tokenizer.encode(r[f], add_special_tokens=False).ids for r in probes
                              for f in ('source_sentence', 'target_sentence')],
                       special['<|eos|>'], special['<|pad|>'], device, precision)
    probe_results = []
    for i, row in enumerate(probes):
        correct, wrong = scores[2*i:2*i+2]
        probe_results.append(dict(id=row['id'], paradigm=row['paradigm'], correct=correct, wrong=wrong,
                                  correct_sum=correct['sum_nll'] < wrong['sum_nll'],
                                  correct_mean=correct['mean_nll'] < wrong['mean_nll']))
    report['rublimp'] = dict(panel_sha256=sha256(probe), pairs=probe_results, summary={})
    for paradigm in ['overall'] + sorted({r['paradigm'] for r in probe_results}):
        subset = [r for r in probe_results if paradigm == 'overall' or r['paradigm'] == paradigm]
        report['rublimp']['summary'][paradigm] = dict(n=len(subset), **{
            'accuracy_' + m: sum(r['correct_' + m] for r in subset)/len(subset) for m in ('sum', 'mean')})
    save('rublimp')
    if args.quick:
        rows = [rows[i] for i in evenly_spaced(len(rows), 16)]
    encoded, null_encoded = [], []
    null_indices = set(evenly_spaced(len(rows), min(100, len(rows))))
    for i, row in enumerate(rows):
        if len(row['negatives']) != 3 or len(set([row['positive']] + row['negatives'])) != 4:
            raise ValueError('Expected four distinct candidates')
        for answer in [row['positive']] + row['negatives']:
            encoded.append(candidate_tokens(tokenizer, row['prefix'], answer, model.config.context_length))
            if i in null_indices:
                null_encoded.append(candidate_tokens(tokenizer, '', answer, model.config.context_length))
    scored = score_candidates(model, encoded, special['<|pad|>'], device, precision)
    null_scored = iter(score_candidates(model, null_encoded, special['<|pad|>'], device, precision))
    results = [candidate_result(row, scored[4*i:4*i+4], [next(null_scored) for _ in range(4)]
                               if i in null_indices else None) for i, row in enumerate(rows)]
    report['causal'] = dict(rows=results, summary=summarize_candidates(results),
                           scoring='Full prefix+answer BPE boundary checked; sum primary, mean sensitivity; strict ties fail')
    texts = [tokenizer.encode(r['text'], add_special_tokens=False).ids for r in rows]
    text_scores = score_ids(model, texts, special['<|eos|>'], special['<|pad|>'], device, precision)
    report['causal']['text_likelihood'] = aggregate_likelihood(sum(s['sum_nll'] for s in text_scores), sum(s['targets'] for s in text_scores))
    save('causal')
    if metrics_only:
        report['complete'] = True
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
        save('complete')
        return
    train_grams, report['copy_screen'] = sampled_training_ngrams(32 if args.quick else 2048)
    for case in panel:
        seed = 7 + int(hashlib.sha256(case['id'].encode()).hexdigest()[:8], 16)
        seed_everything(seed)
        ids, reason = generate(model, encode_continuation(tokenizer, case['prefix']), 96, .7, 40,
                               [special['<|eos|>'], special['<|eot|>']],
                               [special[t] for t in ('<|pad|>', '<|bos|>', '<|user|>', '<|assistant|>')],
                               precision, top_p=.9)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        report['samples'].append(dict(**case, seed=seed, text=text, first_two=first_two_sentences(text),
                                      generated_tokens=len(ids), stop_reason=reason,
                                      **lexical_metrics(text, case['prefix']), **training_copy_fraction(ids, train_grams)))
        if len(report['samples']) % 10 == 0:
            save('generation_' + str(len(report['samples'])))
    report['generation_summary'] = dict(n=len(panel), groups=len({r['group'] for r in panel}),
        stop_reasons=dict(Counter(r['stop_reason'] for r in report['samples'])),
        means={k: sum(r[k] for r in report['samples'] if r[k] is not None)/max(1, sum(r[k] is not None for r in report['samples']))
               for k in ('generated_tokens', 'words', 'distinct2', 'repeated4', 'prompt_copy4', 'sampled_train_copy12')})
    report['complete'] = True
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
    save('complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'reserve'], default='validation')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true')
    mode.add_argument('--metrics-only', action='store_true',
                      help='Full likelihood, RuBLiMP and causal metrics; skip generation and copy screen')
    args = parser.parse_args()
    with run_lock(ROOT/'runs/pretrain_series01', 'series'):
        evaluate(args)


if __name__ == '__main__':
    main()
