"""Read-only minimal-pair teacher-forcing diagnostic; no optimizer or generation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer
from poetrygpt.data import read_json, sha256
from poetrygpt.model import GPT, ModelConfig, token_losses
from poetrygpt.runtime import autocast, load_checkpoint, resolve_device, run_lock, write_json
from scripts.run_pretrain_series import SERIES, LEDGER


@torch.no_grad()
def score_ids(model, sequences, eos_id, pad_id, device, precision='fp32', batch_size=16, deadline=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    model.eval()
    results = []
    for start in range(0, len(sequences), batch_size):
        if deadline and datetime.now(timezone.utc) >= deadline:
            raise RuntimeError('Series deadline reached during diagnostic')
        rows = sequences[start:start + batch_size]
        if any(not row or len(row) + 1 > model.config.context_length for row in rows):
            raise ValueError('Empty/over-context sentence; never silently truncate')
        width = max(len(row) + 1 for row in rows)
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
        labels = torch.full_like(ids, -100)
        for i, row in enumerate(rows):
            ids[i, 0] = eos_id
            ids[i, 1:len(row) + 1] = torch.tensor(row, device=device)
            labels[i, 1:len(row) + 1] = ids[i, 1:len(row) + 1]
        with autocast(device, precision):
            losses, mask = token_losses(model(ids), labels)
        if not torch.isfinite(losses[mask]).all():
            raise FloatingPointError('Nonfinite sentence NLL')
        for i, row in enumerate(rows):
            targets = int(mask[i].sum())
            assert targets == len(row)
            total = float(losses[i].sum())
            results.append({'targets': targets, 'sum_nll': total, 'mean_nll': total / targets})
    return results


def run(checkpoint, output, device_name='cuda'):
    panel = ROOT / 'audit/rublimp_probe/pairs.jsonl'
    manifest = read_json(ROOT / 'audit/rublimp_probe/manifest.json')
    tokenizer_path = ROOT / manifest['tokenizer']
    if sha256(panel) != manifest['panel_sha256'] or sha256(tokenizer_path) != manifest['tokenizer_sha256']:
        raise ValueError('Frozen diagnostic input changed')
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve existing diagnostic result')
    with run_lock(SERIES, 'series'):
        ledger = read_json(LEDGER)
        deadline = datetime.fromisoformat(ledger['deadline_utc'])
        if ledger['status'] != 'active' or datetime.now(timezone.utc) >= deadline:
            raise ValueError('Series is closed')
        before_ledger = sha256(LEDGER)
        state = load_checkpoint(checkpoint)
        if state['tokenizer_sha256'] != manifest['tokenizer_sha256']:
            raise ValueError('Checkpoint tokenizer differs from diagnostic')
        device, precision = resolve_device(device_name, 'auto')
        torch.set_num_threads(4)
        model = GPT(ModelConfig(**state['model_config'])).to(device)
        model.load_state_dict(state['model'])
        del state
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        rows = [json.loads(line) for line in panel.read_text(encoding='utf-8').splitlines()]
        encoded = [tokenizer.encode(row[field], add_special_tokens=False).ids
                   for row in rows for field in ('source_sentence', 'target_sentence')]
        scores = score_ids(model, encoded, tokenizer.token_to_id('<|eos|>'), tokenizer.token_to_id('<|pad|>'),
                           device, precision, deadline=deadline)
        groups = defaultdict(list)
        pairs = []
        for i, row in enumerate(rows):
            source, target = scores[2 * i:2 * i + 2]
            result = {'id': row['id'], 'paradigm': row['paradigm'], 'source': source, 'target': target,
                      'correct_mean': source['mean_nll'] < target['mean_nll'],
                      'correct_sum': source['sum_nll'] < target['sum_nll'],
                      'mean_margin': target['mean_nll'] - source['mean_nll'],
                      'equal_token_length': source['targets'] == target['targets']}
            pairs.append(result)
            groups[row['paradigm']].append(result)
            groups['overall'].append(result)
        summary = {key: {'pairs': len(values),
                        'accuracy_mean': sum(r['correct_mean'] for r in values) / len(values),
                        'accuracy_sum': sum(r['correct_sum'] for r in values) / len(values),
                        'ties_mean': sum(r['mean_margin'] == 0 for r in values),
                        'equal_token_length_pairs': sum(r['equal_token_length'] for r in values)}
                   for key, values in groups.items()}
        if sha256(LEDGER) != before_ledger:
            raise ValueError('Ledger changed during locked diagnostic')
        write_json(output, {'checkpoint_sha256': sha256(checkpoint), 'panel_sha256': sha256(panel),
                           'training_performed': False, 'precision': precision, 'summary': summary, 'pairs': pairs,
                           'scoring': 'true EOS ID prefix, no final EOS; full sentence token mean NLL primary, sum NLL sensitivity; strict less-than, ties reported separately; no sampling or forbidden-token masking',
                           'limits': 'Adapted small diagnostic, not full RuBLiMP or an era benchmark; does not measure free-generation coherence.'})
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    print(run(args.checkpoint, args.output, args.device))
