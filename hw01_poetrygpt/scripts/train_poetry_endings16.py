"""Optional x3 auxiliary ending-word CE, normalized over the complete update.

Frozen units/poetry trainers and streams are unchanged. Ordinary target quotas,
LR, main-poetry labels and validation stay intact; only the training objective
changes. Auxiliary endings are aligned on complete records, never cropped text.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tokenizers import Tokenizer
from poetrygpt.data import PreparedDataset, read_json, sha256
from poetrygpt.model import token_losses
from poetrygpt.runtime import autocast
from scripts import train_poetry_units16 as units
from scripts import train_poetry08 as trainer
from scripts.evaluate_contextual16 import ending_positions
from scripts.poetry16_streams import TargetRecordStream


def align_endings(data, tokenizer):
    """Fail closed on anything except frozen complete second-line targets."""
    if data.manifest.get('serialization') != 'source_context_onlysecondline_newline_no_eos_loss':
        raise ValueError('Expected frozen contextual second-line serialization')
    result = []
    for index, meta in enumerate(data.metadata):
        ids, labels = data.record(index)
        targets = np.flatnonzero(labels != -100)
        if (not len(targets) or targets[0] < 1
                or not np.array_equal(targets, np.arange(targets[0], len(ids)))
                or not np.array_equal(labels[targets], ids[targets])
                or np.isin(ids[targets], list(data.manifest['special_tokens'].values())).any()):
            raise ValueError('Expected contiguous second-line labels without special-token loss')
        if (meta.get('task_type') != 'onlysecondline' or meta.get('target_includes_genuine_newline') is not True
                or meta.get('eos_supervised') is not False or meta.get('context_positions') != int(targets[0])
                or meta.get('positions') != len(ids) or meta.get('target_count') != len(targets)):
            raise ValueError('Second-line metadata/label boundary mismatch')
        # Reuse the diagnostic's strict whole-record token roundtrip and offsets.
        positions = ending_positions(ids, labels, tokenizer)
        text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
        encoded = tokenizer.encode(text, add_special_tokens=False)
        boundary = encoded.offsets[int(targets[0])][0]
        target = text[boundary:]
        if (not target.endswith('\n') or '\n' in target[:-1]
                or not text[:boundary].endswith('\n')
                or hashlib.sha256(target.encode()).hexdigest() != meta.get('target_text_sha256')):
            raise ValueError('Expected exactly one source-hashed target line and genuine newline')
        words = list(re.finditer('[А-Яа-яЁё]+', text))
        start, end = words[-1].span()
        all_positions = [i for i, (a,b) in enumerate(encoded.offsets) if a < end and b > start]
        if start < boundary or positions != all_positions:
            raise ValueError('Full ending word must belong to the supervised second line')
        result.append(positions)
    return result


class EndingBatch(tuple):
    """Still unpacks as (ids, labels); mask is derived from source record IDs."""
    def __new__(cls, ids, labels, endings):
        result = super().__new__(cls, (ids, labels))
        if endings.shape != labels.shape or endings.dtype != torch.bool or (endings & labels.eq(-100)).any():
            raise ValueError('Invalid supervised ending mask')
        result.endings = endings
        return result


class RecordReads:
    """Read-only view recording full-record provenance for the unchanged stream."""
    def __init__(self, data):
        self.original, self.reads = data, None

    def __len__(self):
        return len(self.original)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def record(self, index):
        if self.reads is not None:
            self.reads.append(int(index))
        return self.original.record(index)


class EndingStream(TargetRecordStream):
    def __init__(self, data, seed, state=None, *, endings):
        self.endings = endings
        super().__init__(RecordReads(data), seed, state)

    def next(self, quota, batch_size):
        self.data.reads = []
        try:
            batches = super().next(quota, batch_size)
            reads = self.data.reads
        finally:
            self.data.reads = None
        if len(reads) != sum(len(ids) for ids, _ in batches):
            raise AssertionError('Stream full-record reads no longer match batch rows')
        output, offset = [], 0
        for ids, labels in batches:
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for row in range(len(ids)):
                index = reads[offset]
                offset += 1
                original_ids, original_labels = self.data.original.record(index)
                selected = torch.nonzero(labels[row].ne(-100)).flatten().numpy()
                end = int(selected[-1])+1
                if (not np.array_equal(ids[row,:end].numpy(), original_ids[:end])
                        or not np.array_equal(labels[row,selected].numpy(), original_labels[selected])
                        or ids[row,end:].ne(self.data.pad_id).any()):
                    raise AssertionError('Auxiliary cropped row differs from its full source record')
                for position in self.endings[index]:
                    if position < end and labels[row,position] != -100:
                        mask[row,position] = True
            output.append(EndingBatch(ids, labels, mask))
        return output

    def statistics(self):
        count = 0
        for seen, targets, endings in zip(self.record_targets, self.positions, self.endings):
            complete, partial = divmod(int(seen), len(targets))
            count += complete*len(endings) + int(np.isin(targets[:partial], endings).sum())
        return {**super().statistics(), 'endword_targets': count}


class TaggedMixture:
    def __init__(self, mixed):
        self.mixed = mixed

    def __getattr__(self, name):
        return getattr(self.mixed, name)

    def next(self, quota, batch_size):
        batches, counts = self.mixed.next(quota, batch_size)
        tagged = sum(int(batch[1][:,1:].ne(-100).sum()) for batch in batches if isinstance(batch, EndingBatch))
        if tagged != counts['auxiliary']:
            raise AssertionError('Tagged auxiliary targets differ from mixture domain count')
        return batches, counts


def backward_endings(model, batches, quota, device, precision, weight, ordinary_backward):
    actual = sum(int(labels[:,1:].ne(-100).sum()) for _, labels in batches)
    endings = sum(int(batch.endings[:,1:].sum()) for batch in batches if isinstance(batch, EndingBatch))
    if actual != quota or quota < 1 or type(weight) is not int or weight not in (1,3):
        raise ValueError('Invalid weighted update quota/weight')
    denominator = quota + (weight-1)*endings
    if weight == 1 or endings == 0:
        ordinary = ordinary_backward(model, batches, quota, device, precision)
        objective = ordinary
    else:
        ordinary_sum, weighted_sum = 0., 0.
        for batch in batches:
            ids, labels = (value.to(device) for value in batch)
            with autocast(device, precision):
                losses, _ = token_losses(model(ids), labels)
                summed = losses.sum()
                weighted = losses
                if isinstance(batch, EndingBatch):
                    weighted = losses * (1 + (weight-1)*batch.endings[:,1:].to(device))
                weighted = weighted.sum()
            if not torch.isfinite(summed) or not torch.isfinite(weighted):
                raise FloatingPointError('Nonfinite weighted training objective')
            (weighted/denominator).backward()
            ordinary_sum += summed.detach().item()
            weighted_sum += weighted.detach().item()
        ordinary, objective = ordinary_sum/quota, weighted_sum/denominator
    return ordinary, dict(ordinary_nll=ordinary, weighted_objective=objective,
        sum_weights=denominator, endword_targets=endings, ending_weight=weight)


@contextmanager
def experiment_context(raw):
    if set(raw) != {'units','ending_weight'} or type(raw['ending_weight']) is not int or raw['ending_weight'] not in (1,3):
        raise ValueError('Expected units recipe and ending_weight 1 or 3')
    recipe, weight = raw['units'], raw['ending_weight']
    cfg = units.configuration(recipe)
    if recipe['auxiliary'] is None:
        raise ValueError('Ending objective requires the explicit auxiliary stream')
    data = PreparedDataset(units.checked(recipe['auxiliary']['manifest']).parent,
        'sft', 'train', cfg['model']['context_length'])
    endings = align_endings(data, Tokenizer.from_file(cfg['tokenizer']))
    alignment_sha = hashlib.sha256(json.dumps(endings).encode()).hexdigest()
    old_target_stream = units.TargetRecordStream
    with units.experiment_context(recipe) as cfg:
        old_identity, old_stream = trainer.input_identity, trainer.MixtureStream
        old_backward, old_event = trainer.backward_microbatches, trainer.append_event
        last_update = None

        def auxiliary(dataset, seed, state=None):
            if dataset.hashes != data.hashes:
                raise ValueError('Auxiliary changed after ending alignment')
            return EndingStream(dataset, seed, state, endings=endings)

        def identity(*args, **kwargs):
            result = old_identity(*args, **kwargs)
            result['poetry16_endings'] = dict(recipe=deepcopy(raw), alignment_sha256=alignment_sha,
                wrapper_sha256=sha256(__file__), evaluator_sha256=sha256(ROOT/'scripts/evaluate_contextual16.py'),
                objective='sum(w*CE)/sum(w) across entire update; auxiliary final Cyrillic word only',
                quotas='unweighted supervised targets; all main and auxiliary labels retained',
                token_overlap='Every subtoken overlapping the complete final word; token may include adjacent punctuation')
            return result

        def stream(*args, **kwargs):
            return TaggedMixture(old_stream(*args, **kwargs))

        def backward(model, batches, count, device, precision):
            nonlocal last_update
            ordinary, last_update = backward_endings(model, batches, count, device, precision, weight, old_backward)
            return ordinary

        def event(path, value):
            nonlocal last_update
            if value.get('event') == 'train':
                if (last_update is None or last_update['ordinary_nll'] != value['nll']
                        or last_update['endword_targets'] > value['domain_targets']['auxiliary']
                        or last_update['sum_weights'] != value['supervised_targets']+(weight-1)*last_update['endword_targets']):
                    raise AssertionError('Weighted logging/update mismatch')
                value = {**value, **last_update}
                last_update = None
            old_event(path, value)

        units.TargetRecordStream = auxiliary
        trainer.input_identity, trainer.MixtureStream = identity, stream
        trainer.backward_microbatches, trainer.append_event = backward, event
        try:
            yield cfg
        finally:
            units.TargetRecordStream = old_target_stream
            trainer.input_identity, trainer.MixtureStream = old_identity, old_stream
            trainer.backward_microbatches, trainer.append_event = old_backward, old_event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--until-tokens', type=int, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.device == 'cuda':
        units.shared.AUDIT = ROOT/'audit/poetry16/gpu_admission.jsonl'
        trainer.check_gpu_idle = lambda: units.shared.admit(torch, 10240,
            units.shared.Audit(coordinator_sha256=sha256(__file__), phase='poetry16_endings'))
    with experiment_context(read_json(args.config)) as cfg:
        trainer.train(cfg, args.run_dir, args.until_tokens, args.device, args.resume, args.stop_after)


if __name__ == '__main__':
    main()
