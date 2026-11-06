"""Training-only ending plans from frozen contextual records; no oracle dev input.

The existing units adapter can consume this ordinary masked-CE dataset directly.
This is a separate continuation protocol, never a replacement for instruction SFT.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tokenizers import Tokenizer
from poetrygpt.data import PreparedDataset, read_json, sha256
from poetrygpt.serialization import pad_batch, special_ids
from scripts.train_poetry_endings16 import align_endings

SERIALIZATION = 'context_ending_plan_then_source_line_v1'
PLAN_MARKER = 'Рифма:\n'
LINE_MARKER = '\n'
WORD = re.compile('[А-Яа-яЁё]+')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def input_key(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def source_parts(ids, labels, tokenizer):
    """Caller first validates complete source records with align_endings()."""
    start = int(np.flatnonzero(labels != -100)[0])
    context_ids, line_ids = ids[:start].tolist(), ids[start:].tolist()
    context = tokenizer.decode(context_ids, skip_special_tokens=False)
    line = tokenizer.decode(line_ids, skip_special_tokens=False)
    if (not context.endswith('\n') or not line.endswith('\n') or '\n' in line[:-1]
            or tokenizer.encode(context + line, add_special_tokens=False).ids != ids.tolist()):
        raise ValueError('Source context/line boundary failed roundtrip')
    return context_ids, line_ids, context, line


def encode_plan(context_ids, line_ids, context, line, tokenizer, context_length=512):
    words = WORD.findall(line)
    if not words:
        raise ValueError('Missing source ending')
    ending = words[-1]
    prefix_text = context + PLAN_MARKER
    plan_text = ending + LINE_MARKER
    text = prefix_text + plan_text + line
    encoded = tokenizer.encode(text, add_special_tokens=False)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False).ids
    before_line = tokenizer.encode(prefix_text + plan_text, add_special_tokens=False).ids
    if (encoded.ids[:len(prefix_ids)] != prefix_ids
            or encoded.ids[:len(before_line)] != before_line
            or prefix_ids[:len(context_ids)] != context_ids
            or encoded.ids[len(before_line):] != line_ids
            or tokenizer.decode(encoded.ids, skip_special_tokens=False) != text):
        raise ValueError('Protocol boundary retokenized original context/line or plan prefix')
    if len(encoded.ids) > context_length:
        raise ValueError('Planning record exceeds context; cropping forbidden')
    target_start, line_start = len(prefix_ids), len(before_line)
    labels = [-100] * target_start + encoded.ids[target_start:]
    if any(token in special_ids(tokenizer).values() for token in encoded.ids[target_start:]):
        raise ValueError('Unexpected special-token training target')
    # Token categories partition the actual quota. A BPE token overlapping the
    # final word belongs to plan_word, including any boundary whitespace.
    word_end = len(prefix_text) + len(ending)
    word_count = sum(a < word_end and b > len(prefix_text)
                     for a, b in encoded.offsets[target_start:line_start])
    counts = dict(plan_word_targets=word_count,
                  protocol_targets=line_start-target_start-word_count,
                  line_targets=len(line_ids), supervised_targets=len(encoded.ids)-target_start,
                  masked_prefix_positions=target_start, positions=len(encoded.ids))
    record = dict(input_ids=encoded.ids, labels=labels, attention_mask=[1]*len(encoded.ids))
    return record, dict(ending=ending, line_start=line_start, **counts)


def visible_case(meta, context_ids, context, tokenizer):
    visible = tokenizer.decode(context_ids, skip_special_tokens=True)
    lines = [line for line in visible.splitlines() if line.strip()]
    if not lines or not context.endswith('\n'):
        raise ValueError('Missing complete visible first line')
    return dict(id=meta['id'], group_id=meta['family_id'], context_ids=context_ids,
                context_text=context, first_line=lines[-1])


def exposure(data, state):
    """Exact plan/protocol/line target counts at any saved auxiliary cursor."""
    from scripts.poetry16_streams import TargetRecordStream
    stream = TargetRecordStream(data, state['seed'], state)
    counts = Counter()
    categories = ('plan_word_targets', 'protocol_targets', 'line_targets')
    for meta, seen in zip(data.metadata, stream.record_targets):
        plan = meta['plan']
        complete, partial = divmod(int(seen), plan['supervised_targets'])
        for key in categories:
            taken = min(partial, plan[key])
            counts[key] += complete*plan[key] + taken
            partial -= taken
    if sum(counts.values()) != stream.total_targets:
        raise AssertionError('Exposure categories disagree with saved quota')
    return dict(supervised_targets=stream.total_targets, **counts)


def build(source, tokenizer_path, output):
    source, tokenizer_path, output = map(Path, (source, tokenizer_path, output))
    if output.exists():
        raise FileExistsError(output)
    original = read_json(source/'manifest.json')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    token_hash = sha256(tokenizer_path)
    if (token_hash not in [v for k,v in original['inputs_sha256'].items()
                          if k.replace('\\', '/').endswith('tokenizer.json')]
            or original['special_tokens'] != special_ids(tokenizer)):
        raise ValueError('Source tokenizer provenance mismatch')
    datasets = {split: PreparedDataset(source, 'sft', split, original['context_length'])
                for split in ('train', 'validation')}
    families = [{row['family_id'] for row in data.metadata} for data in datasets.values()]
    if families[0] & families[1]:
        raise ValueError('Train/validation family overlap')
    inputs = {input_key(source/'manifest.json'): sha256(source/'manifest.json'),
              input_key(tokenizer_path): token_hash, input_key(__file__): sha256(__file__),
              'scripts/train_poetry_endings16.py': sha256(ROOT/'scripts/train_poetry_endings16.py'),
              'scripts/evaluate_contextual16.py': sha256(ROOT/'scripts/evaluate_contextual16.py')}
    records, metadata, cases, total = [], [], [], Counter()
    for split, data in datasets.items():
        align_endings(data, tokenizer)
        inputs.update({input_key(source/name): digest for name,digest in data.hashes.items()})
        for index, meta in enumerate(data.metadata):
            ids, labels = data.record(index)
            context_ids, line_ids, context, line = source_parts(ids, labels, tokenizer)
            if split == 'validation':
                cases.append(visible_case(meta, context_ids, context, tokenizer))
                continue  # Never derive or expose a validation ending plan.
            record, plan = encode_plan(context_ids, line_ids, context, line, tokenizer, data.context_length)
            if plan['ending'].lower().replace('ё','е') != meta['ending_words'][1].lower().replace('ё','е'):
                raise ValueError('Source ending metadata disagrees')
            records.append(record)
            metadata.append(dict(id=meta['id'], group_id=meta['family_id'], split='train',
                task_type='ending_plan_secondline', source_metadata=deepcopy(meta), plan=plan))
            total.update({key: plan[key] for key in ('plan_word_targets', 'protocol_targets',
                'line_targets', 'supervised_targets', 'masked_prefix_positions')})
    if total['supervised_targets'] != sum(total[k] for k in ('plan_word_targets','protocol_targets','line_targets')):
        raise AssertionError('Target budget does not partition')
    if any(sha256(ROOT/path) != digest for path,digest in inputs.items()):
        raise ValueError('Input changed during build')
    output.mkdir(parents=True)
    padded = pad_batch(records, original['special_tokens']['<|pad|>'])
    np.savez_compressed(output/'sft_train.npz', **{k: np.asarray(v, dtype=np.int32) for k,v in padded.items()})
    write_json(output/'sft_train_index.json', metadata)
    for name in ('sft_validation.npz', 'sft_validation_index.json'):
        shutil.copyfile(source/name, output/name)
        if sha256(output/name) != sha256(source/name):
            raise AssertionError('Frozen validation copy changed')
    write_json(output/'dev_prefixes.json', dict(schema='ending-plan16-visible-prefixes-1',
        source_manifest_sha256=inputs[input_key(source/'manifest.json')], cases=cases))
    manifest = dict(release=output.name, serialization=SERIALIZATION,
        context_length=original['context_length'], special_tokens=original['special_tokens'],
        dataset_role='auxiliary planned continuation only; not instruction SFT',
        inputs_sha256=inputs, source_manifest_sha256=inputs[input_key(source/'manifest.json')],
        protocol=dict(plan_marker=PLAN_MARKER, line_marker=LINE_MARKER, eos_targets=0,
            prefix_masked=True, original_context_and_line_token_ids_unchanged=True),
        train=dict(records=len(records), maximum_positions=max(len(r['input_ids']) for r in records), **total),
        validation=dict(records=len(cases), serialization=original['serialization'],
            byte_identical_to_source=True, role='ordinary CE control; not plan teacher forcing'),
        evaluation=dict(dev='dev_prefixes.json', oracle=False,
            metrics='format; own-plan compliance; actual line rhyme; meaning requires separate reading'),
        training_performed=False, output_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'manifest.json', manifest)
    PreparedDataset(output, 'sft', 'train', original['context_length'])
    PreparedDataset(output, 'sft', 'validation', original['context_length'])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source, args.tokenizer, args.output)
    print(json.dumps(result['train']))


if __name__ == '__main__':
    main()
