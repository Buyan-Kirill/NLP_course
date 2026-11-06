"""Recover source windows and masked ending-plan examples from poetry IDs."""
import json
import numpy as np
from tokenizers import Tokenizer

from reproduction.datasets import catalog, restore_sidecar
from reproduction.download import ROOT, sha256
from poetrygpt.serialization import pad_batch
from scripts.prepare_ending_plan16 import encode_plan, visible_case
from reproduction.array_archive import write_npz


def verify_outputs(folder):
    manifest = json.loads((folder / 'manifest.json').read_text('utf8'))
    for name, expected in manifest['output_sha256'].items():
        if sha256(folder / name) != expected:
            raise ValueError('Auxiliary reconstruction differs: ' + str(folder / name))


def prepare():
    for relative, item in catalog()['auxiliary'].items():
        restore_sidecar(relative, item['sha256'], item['compressed'])
    windows = ROOT / 'data/poetry16_units'
    planned = ROOT / 'data/continuation16_ending_plan'
    source = ROOT / 'data/prepared_v7/poetry_supplement'
    tokenizer = Tokenizer.from_file(str(ROOT / 'artifacts/tokenizers_v3/selected/tokenizer.json'))
    cases = []
    for split in ['train', 'validation']:
        window_meta = json.loads((windows / f'poetry_{split}_windows_metadata.json').read_text('utf8'))
        array = np.asarray([[r['input_start'], r['end'], r['owner']] for r in window_meta], dtype=np.int64)
        window_path = windows / f'poetry_{split}_windows.npy'
        if not window_path.exists():
            np.save(window_path, array, allow_pickle=False)
        ids = np.load(source / f'poetry_{split}.npy', mmap_mode='r', allow_pickle=False)
        metadata = json.loads((planned / f'sft_{split}_index.json').read_text('utf8'))
        records = []
        for row in metadata:
            meta = row['source_metadata'] if split == 'train' else row
            a, b = meta['source_token_context']
            c, d = meta['source_token_target']
            context_ids = meta['actual_work_boundary_prefix'] + ids[a:b].tolist()
            line_ids = ids[c:d].tolist()
            context = tokenizer.decode(context_ids, skip_special_tokens=False)
            line = tokenizer.decode(line_ids, skip_special_tokens=False)
            if split == 'train':
                record, plan = encode_plan(context_ids, line_ids, context, line, tokenizer)
                if plan != row['plan']:
                    raise ValueError('Ending-plan token boundaries differ: ' + row['id'])
            else:
                tokens = context_ids + line_ids
                record = dict(input_ids=tokens, labels=[-100]*len(context_ids)+line_ids,
                              attention_mask=[1]*len(tokens))
                cases.append(visible_case(meta, context_ids, context, tokenizer))
            records.append(record)
        padded = pad_batch(records, 0, length=None if split == 'train' else 512)
        dtypes = dict(input_ids=np.int32, labels=np.int32, attention_mask=np.int32) if split == 'train' else dict(input_ids=np.uint16, labels=np.int32, attention_mask=np.uint8)
        path = planned / f'sft_{split}.npz'
        if not path.exists():
            write_npz(path, {key: np.asarray(value, dtype=dtypes[key]) for key, value in padded.items()})
    manifest = json.loads((planned / 'manifest.json').read_text('utf8'))
    prefixes = dict(schema='ending-plan16-visible-prefixes-1',
                    source_manifest_sha256=manifest['source_manifest_sha256'], cases=cases)
    path = planned / 'dev_prefixes.json'
    if not path.exists():
        ending = catalog()['serialized_formats'][path.relative_to(ROOT).as_posix()]['newline']
        text = json.dumps(prefixes, ensure_ascii=False, indent=2)+'\n'
        path.write_bytes(text.replace('\n', ending).encode('utf8'))
    verify_outputs(windows)
    verify_outputs(planned)
    print('Poetry windows and ending-plan arrays: original SHA256 verified', flush=True)
