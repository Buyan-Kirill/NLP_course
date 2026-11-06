"""Read prepared_v3 arrays with deterministic complete-epoch sampling."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class PreparedDataset:
    def __init__(self, folder, domain, split, context_length=512):
        if domain not in ('prose', 'poetry', 'sft') or split not in ('train', 'validation', 'test'):
            raise ValueError('Unknown dataset domain/split')
        self.folder, self.domain, self.split = Path(folder), domain, split
        self.context_length = context_length
        self.manifest = read_json(self.folder / 'manifest.json')
        if self.manifest['context_length'] != context_length:
            raise ValueError('Context must match the frozen preparation contract')
        self.pad_id = self.manifest['special_tokens']['<|pad|>']
        suffix = '.npz' if domain == 'sft' else '.npy'
        path = self.folder / f'{domain}_{split}{suffix}'
        index_path = self.folder / f'{domain}_{split}_{"index" if domain == "sft" else "offsets"}.json'
        self.hashes = {p.name: sha256(p) for p in (path, index_path)}
        for name, digest in self.hashes.items():
            if self.manifest['output_sha256'][name] != digest:
                raise ValueError(f'Prepared input hash mismatch: {name}')
        self.metadata = read_json(index_path)
        if domain == 'sft':
            with np.load(path, allow_pickle=False) as archive:
                self.arrays = {key: archive[key] for key in ('input_ids', 'labels', 'attention_mask')}
            mask = self.arrays['attention_mask']
            if not np.isin(mask, [0, 1]).all() or np.any(np.diff(mask.astype(np.int16), axis=1) > 0):
                raise ValueError('Only contiguous right padding is supported')
            if np.any(self.arrays['labels'][mask == 0] != -100):
                raise ValueError('Padding must not be supervised')
            self.length = len(mask)
        else:
            self.ids = np.load(path, mmap_mode='r', allow_pickle=False)
            self.length = math.ceil((len(self.ids) - 1) / (context_length - 1))
            self.ends = np.asarray([r['end'] for r in self.metadata], dtype=np.int64)
        if self.length < 1:
            raise ValueError('Empty training/evaluation dataset')

    def __len__(self):
        return self.length

    def record(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.domain == 'sft':
            n = int(self.arrays['attention_mask'][index].sum())
            return self.arrays['input_ids'][index, :n].astype(np.int64), self.arrays['labels'][index, :n].astype(np.int64)
        start = index * (self.context_length - 1)
        ids = np.array(self.ids[start:start + self.context_length], dtype=np.int64)
        labels = ids.copy()
        labels[0] = -100
        return ids, labels

    def batch(self, indices, device):
        records = [self.record(int(i)) for i in indices]
        width = max(len(ids) for ids, _ in records)
        ids = torch.full((len(records), width), self.pad_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        for row, (tokens, targets) in enumerate(records):
            ids[row, :len(tokens)] = torch.from_numpy(tokens)
            labels[row, :len(tokens)] = torch.from_numpy(targets)
        return ids.to(device), labels.to(device)

    def target_owners(self, index):
        """Record index for each target (including EOS), used by sliced eval."""
        ids, labels = self.record(index)
        if self.domain == 'sft':
            return np.full(len(ids) - 1, index), labels[1:] != -100
        start = index * (self.context_length - 1)
        positions = np.arange(start + 1, start + len(ids))
        return np.searchsorted(self.ends, positions, side='right'), np.ones(len(positions), dtype=bool)


class BatchCursor:
    """No dropped tail batch; local RNG does not consume model/dropout RNG."""
    def __init__(self, size, batch_size, seed, epoch=0, offset=0):
        if size < 1 or batch_size < 1 or epoch < 0 or not 0 <= offset <= size:
            raise ValueError('Invalid sampler state')
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.offset = epoch, offset
        self._order = None

    def next(self):
        if self.offset == self.size:
            self.epoch, self.offset, self._order = self.epoch + 1, 0, None
        if self._order is None:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            self._order = torch.randperm(self.size, generator=generator).tolist()
        stop = min(self.size, self.offset + self.batch_size)
        indices = self._order[self.offset:stop]
        self.offset = stop
        return indices

    def state_dict(self):
        return {'epoch': self.epoch, 'offset': self.offset}
