"""Teacher-forced NLL, per-family macro NLL and metadata slices; test is explicit."""
import argparse
from collections import defaultdict
import math
from pathlib import Path

import numpy as np
import torch

from .data import PreparedDataset, sha256
from .model import GPT, ModelConfig, token_losses
from .runtime import autocast, load_checkpoint, resolve_device, write_json


def slice_keys(row, domain):
    keys = {'overall', f'domain:{domain}', f"author:{row.get('author') or 'unknown'}"}
    for field in ('genre', 'topic', 'topics', 'themes', 'task', 'knowledge_mode', 'modern_context',
                  'date_basis', 'source_genre', 'orthography', 'core_style'):
        value = row.get(field)
        values = value if isinstance(value, list) else [value] if value is not None else ['unknown']
        keys.update(f'{field}:{item}' for item in values or ['unknown'])
    for source in row.get('sources') or []:
        keys.add('source:' + (source.get('dataset', 'unknown') if isinstance(source, dict) else source))
    for field in ('meter', 'feet'):
        keys.add(f"verse_{field}:{(row.get('verse') or {}).get(field) or 'unknown'}")
    dates = row.get('dates') or {}
    year = dates.get('written_from', row.get('written_from'))
    last_year = dates.get('written_to', row.get('written_to'))
    if isinstance(year, int) and isinstance(last_year, int) and year // 50 == last_year // 50:
        keys.add(f'written_period:{year // 50 * 50}-{year // 50 * 50 + 49}')
    else:
        keys.add('written_period:unknown_or_cross_period')
    size = row.get('positions', row.get('end', 0) - row.get('start', 0))
    keys.add('length:' + ('<128' if size < 128 else '128-511' if size < 512 else '>=512'))
    return keys


@torch.no_grad()
def evaluate(model, dataset, device, precision='fp32', batch_size=8, max_batches=0):
    if batch_size < 1 or max_batches < 0:
        raise ValueError('Invalid evaluation batch limits')
    was_training = model.training
    model.eval()
    count = min(len(dataset), max_batches * batch_size) if max_batches else len(dataset)
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64)
    sums, counts = np.zeros(len(dataset.metadata), dtype=np.float64), np.zeros(len(dataset.metadata), dtype=np.int64)
    try:
        for offset in range(0, count, batch_size):
            batch_indices = indices[offset:offset + batch_size]
            ids, labels = dataset.batch(batch_indices, device)
            with autocast(device, precision):
                losses, mask = token_losses(model(ids), labels)
            if not torch.isfinite(losses[mask]).all():
                raise FloatingPointError('Nonfinite validation loss')
            values = losses.cpu().numpy()
            for row, index in enumerate(batch_indices):
                owners, valid = dataset.target_owners(int(index))
                np.add.at(sums, owners[valid], values[row, :len(owners)][valid])
                np.add.at(counts, owners[valid], 1)
    finally:
        model.train(was_training)
    records, groups = [], defaultdict(list)
    for index in np.flatnonzero(counts):
        meta = dataset.metadata[index]
        identifier = meta.get('work_id', meta.get('id'))
        family = meta.get('family_id', meta.get('group_id', identifier))
        record = {'id': identifier, 'family_id': family, 'nll_sum': float(sums[index]), 'targets': int(counts[index])}
        records.append(record)
        for key in slice_keys(meta, dataset.domain):
            groups[key].append(record)
    metrics = {}
    for key, members in groups.items():
        total_nll, total_count = sum(x['nll_sum'] for x in members), sum(x['targets'] for x in members)
        families = defaultdict(lambda: [0.0, 0])
        for row in members:
            families[row['family_id']][0] += row['nll_sum']
            families[row['family_id']][1] += row['targets']
        nll = total_nll / total_count
        metrics[key] = {'nll': nll, 'perplexity': math.exp(nll) if nll < 700 else None,
                        'targets': total_count, 'records': len(members), 'families': len(families),
                        'family_macro_nll': sum(s / n for s, n in families.values()) / len(families)}
    return {'split': dataset.split, 'domain': dataset.domain, 'windows': count,
            'coverage': 'full' if count == len(dataset) else 'fixed_evenly_spaced_window_subset',
            'metrics': metrics, 'per_record': records,
            'limits': 'Teacher-forced loss is not generation quality. Partial-window slices are descriptive; multilabel slices overlap.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/prepared_v3'))
    parser.add_argument('--domain', choices=['prose', 'poetry', 'sft'], required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = load_checkpoint(args.checkpoint)
    device, precision = resolve_device(args.device, args.precision)
    dataset = PreparedDataset(args.data, args.domain, args.split, state['model_config']['context_length'])
    if sha256(args.data / 'manifest.json') != state['prepared_manifest_sha256']:
        raise ValueError('Evaluation dataset release differs from checkpoint')
    model = GPT(ModelConfig(**state['model_config'])).to(device)
    model.load_state_dict(state['model'])
    result = evaluate(model, dataset, device, precision, args.batch_size)
    result['checkpoint_sha256'] = sha256(args.checkpoint)
    write_json(args.output, result)
    print(result['metrics']['overall'])


if __name__ == '__main__':
    main()
