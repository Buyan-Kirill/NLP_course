"""Encode recovered texts and verify the original arrays and record offsets."""
import json
import os
from pathlib import Path
from tokenizers import Tokenizer
from reproduction.download import ROOT, sha256
from reproduction.serialize import prepare_pretrain
from reproduction import serialize


class JsonlRows:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open(encoding='utf8') as stream:
            self.count = sum(bool(line.strip()) for line in stream)

    def __len__(self):
        return self.count

    def __iter__(self):
        with self.path.open(encoding='utf8') as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def prepare(domain, output):
    os.environ.setdefault('RAYON_NUM_THREADS', str(min(8, os.cpu_count() or 1)))
    formats = json.loads((ROOT / 'reproduction/corpus/manifest.json').read_text('utf8'))['serialized_formats']
    serialize.LINE_ENDINGS = {Path(name).name: item['newline'] for name, item in formats.items()}
    source = ROOT / ('data/prose_v7' if domain == 'prose' else 'data/poetry_v7')
    frozen = json.loads((output / 'manifest.json').read_text('utf8'))
    tokenizer_path = ROOT / 'artifacts/tokenizers_v3/selected/tokenizer.json'
    expected = frozen['inputs_sha256'][tokenizer_path.relative_to(ROOT).as_posix()]
    if sha256(tokenizer_path) != expected:
        raise ValueError('Selected tokenizer differs from the prepared-data contract')
    metadata = {r['work_id']: r for r in JsonlRows(source / 'metadata.jsonl')}
    if domain == 'poetry':
        for row in JsonlRows(source / 'corpus.jsonl'):
            metadata[row['work_id']] = {**{k: v for k, v in row.items() if k != 'text'},
                                        **metadata[row['work_id']]}
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for split in ['train', 'validation', 'test']:
        filenames = [f'{domain}_{split}.npy', f'{domain}_{split}_offsets.json']
        if all((output / f).exists() and sha256(output / f) == frozen['output_sha256'][f]
               for f in filenames):
            continue
        if any((output / f).exists() for f in filenames):
            raise FileExistsError(f'Incomplete or modified {domain}/{split}; preserve or move these two outputs first')
        filename = 'expanded_train.jsonl' if domain == 'prose' and split == 'train' else split + '.jsonl'
        path = source / filename
        if sha256(path) != frozen['inputs_sha256'][path.relative_to(ROOT).as_posix()]:
            raise ValueError('Recovered JSONL differs from the frozen training input')
        print(f'Encode {domain}/{split}', flush=True)
        stats, _ = prepare_pretrain(tokenizer, JsonlRows(path), metadata, domain, split, out=output)
        if stats != frozen['splits'][split][domain]:
            raise ValueError('Serialization statistics differ')
        for name in filenames:
            if sha256(output / name) != frozen['output_sha256'][name]:
                raise ValueError('Serialization checksum differs: ' + name)
    print(f'{domain}: original array and offset checksums verified', flush=True)
