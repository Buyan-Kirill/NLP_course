"""Exact target quotas for frozen poetry windows and masked auxiliary records.

All cursors use local generators. Stream state contains only primitive values;
no model/dropout RNG is consumed and no token or EOS is manufactured.
"""
from copy import deepcopy
import hashlib

import numpy as np
import torch

from poetrygpt.data import BatchCursor
from scripts.train_causal_series import padded
from scripts.train_pretrain08 import EpochStream


def _positive(value, name, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f'Invalid {name}: expected integer')


def _check_state_schema(state, keys, identity):
    if not isinstance(state, dict) or set(state) != keys:
        raise ValueError('Invalid stream state fields')
    if any(state[key] != value for key, value in identity.items()):
        raise ValueError('Stream identity changed: data, windows or seed')
    cursor = state['cursor']
    if not isinstance(cursor, dict) or set(cursor) != {'epoch', 'offset'}:
        raise ValueError('Invalid cursor state')
    for name in ('epoch', 'offset'):
        _positive(cursor[name], name, allow_zero=True)
    _positive(state['total_targets'], 'total_targets', allow_zero=True)
    if not isinstance(state['record_targets'], list) or any(
            type(n) is not int or n < 0 for n in state['record_targets']):
        raise ValueError('Invalid record target counts')
    pending = state['pending']
    if pending is not None:
        if not isinstance(pending, dict) or set(pending) != {'index', 'consumed'}:
            raise ValueError('Invalid pending state')
        _positive(pending['index'], 'pending index', allow_zero=True)
        _positive(pending['consumed'], 'pending consumed')


def _check_progress(stream, weights, owners):
    """Reconstruct exact coverage from epoch, shuffled cursor and pending tail."""
    cursor = stream.cursor
    if cursor.offset > len(weights) or len(stream.record_targets) != len(stream.data.metadata):
        raise ValueError('Invalid cursor/count dimensions')
    per_record = np.zeros(len(stream.data.metadata), dtype=np.int64)
    np.add.at(per_record, owners, weights)
    expected = per_record * cursor.epoch
    order = torch.randperm(len(weights), generator=torch.Generator().manual_seed(
        stream.seed + cursor.epoch)).tolist()
    completed = cursor.offset
    if stream.pending is not None:
        index, consumed = stream.pending['index'], stream.pending['consumed']
        if (completed == 0 or index != order[completed - 1]
                or not 0 < consumed < int(weights[index])):
            raise ValueError('Pending state disagrees with cursor or target count')
        completed -= 1
        expected[owners[index]] += consumed
    indices = order[:completed]
    np.add.at(expected, owners[indices], weights[indices])
    if (not np.array_equal(expected, stream.record_targets)
            or int(expected.sum()) != stream.total_targets):
        raise ValueError('Stream coverage counts disagree with cursor/pending')


class TargetRecordStream:
    """Shuffle SFT records, counting only their existing labels after position 0.

    A split record replays its complete earlier input context with those earlier
    targets masked. Interspersed user/prefix masks stay masked. Trailing masked
    tokens need no forward pass, and records with no supervised target are invalid.
    """
    def __init__(self, data, seed, state=None):
        _positive(seed, 'seed', allow_zero=True)
        if data.domain != 'sft' or len(data) != len(data.metadata) or len(data) < 1:
            raise ValueError('Expected nonempty SFT records with aligned metadata')
        self.data, self.seed = data, seed
        self.positions = []
        digest = hashlib.sha256()
        for index in range(len(data)):
            ids, labels = data.record(index)
            ids, labels = np.asarray(ids), np.asarray(labels)
            if (ids.ndim != 1 or labels.shape != ids.shape or not 2 <= len(ids) <= data.context_length
                    or ids.dtype.kind not in 'iu' or labels.dtype.kind not in 'iu'
                    or np.any(ids < 0) or labels[0] != -100):
                raise ValueError('Malformed auxiliary record or causal position-zero mask')
            positions = np.flatnonzero(labels != -100)
            if not len(positions) or not np.array_equal(labels[positions], ids[positions]):
                raise ValueError('Auxiliary record needs valid causal supervised targets')
            self.positions.append(positions)
            digest.update(np.asarray([len(ids)], dtype='<i8').tobytes())
            digest.update(np.asarray(ids, dtype='<i8').tobytes())
            digest.update(np.asarray(labels, dtype='<i8').tobytes())
        self.weights = np.asarray([len(p) for p in self.positions], dtype=np.int64)
        self.identity = dict(kind='target_record_v1', seed=seed, data_sha256=digest.hexdigest())
        self.total_targets, self.pending = 0, None
        self.record_targets = np.zeros(len(data), dtype=np.int64)
        self.cursor = BatchCursor(len(data), 1, seed)
        if state is not None:
            _check_state_schema(state, set(self.state()), self.identity)
            self.cursor = BatchCursor(len(data), 1, seed, **state['cursor'])
            self.pending = deepcopy(state['pending'])
            self.total_targets = state['total_targets']
            self.record_targets = np.asarray(state['record_targets'], dtype=np.int64)
            _check_progress(self, self.weights, np.arange(len(data)))

    def state(self):
        return dict(**self.identity, cursor=self.cursor.state_dict(), pending=deepcopy(self.pending),
                    total_targets=self.total_targets, record_targets=self.record_targets.tolist())

    def statistics(self):
        return dict(targets=self.total_targets, epoch=self.cursor.epoch,
                    epoch_offset=self.cursor.offset, records_in_epoch=len(self.data),
                    targets_per_epoch=int(self.weights.sum()),
                    records_seen=int(np.count_nonzero(self.record_targets)), pending=deepcopy(self.pending))

    def next(self, quota, batch_size):
        _positive(quota, 'quota')
        _positive(batch_size, 'batch_size')
        records = []
        while quota:
            if self.pending is None:
                self.pending = dict(index=self.cursor.next()[0], consumed=0)
            index, consumed = self.pending['index'], self.pending['consumed']
            positions = self.positions[index]
            count = min(quota, len(positions) - consumed)
            selected = positions[consumed:consumed + count]
            ids, original_labels = self.data.record(index)
            end = int(selected[-1]) + 1
            labels = np.full(end, -100, dtype=np.int64)
            labels[selected] = original_labels[selected]
            records.append((ids[:end].tolist(), labels.tolist()))
            self.total_targets += count
            self.record_targets[index] += count
            self.pending['consumed'] += count
            quota -= count
            if self.pending['consumed'] == len(positions):
                self.pending = None
        return [padded(records[i:i + batch_size], self.data.pad_id)
                for i in range(0, len(records), batch_size)]


def _validated_windows(data, windows):
    raw = np.asarray(windows)
    if (raw.ndim != 2 or raw.shape[1:] != (3,) or not len(raw)
            or raw.dtype.kind not in 'iu' or len(data.ids) < 2):
        raise ValueError('Frozen windows must be a nonempty integer N x 3 array')
    result = np.array(raw, dtype=np.int64, copy=True)
    expected_start = 1
    for record in data.metadata:
        start, end = record['start'], record['end']
        if (type(start) is not int or type(end) is not int
                or start != expected_start or end <= start or end > len(data.ids)):
            raise ValueError('Poetry metadata must partition original target positions')
        expected_start = end
    if expected_start != len(data.ids):
        raise ValueError('Poetry metadata does not cover original targets')
    for start, end, owner in result:
        if not 0 <= owner < len(data.metadata) or not 0 <= start < end <= len(data.ids):
            raise ValueError('Frozen window bounds or owner are invalid')
        record = data.metadata[owner]
        if (not 2 <= end - start <= data.context_length
                or start + 1 < record['start'] or end > record['end']):
            raise ValueError('Frozen window exceeds context or crosses record ownership')
    expected_start = 1
    for start, end, _ in result[np.argsort(result[:, 0], kind='stable')]:
        if start + 1 != expected_start:
            raise ValueError('Frozen windows must cover every target exactly once')
        expected_start = int(end)
    if expected_start != len(data.ids):
        raise ValueError('Frozen windows leave uncovered targets')
    result.flags.writeable = False
    return result


class UnitWindowStream(EpochStream):
    """EpochStream numerical path with validated, immutable window boundaries."""
    def __init__(self, data, seed, state=None, *, windows):
        _positive(seed, 'seed', allow_zero=True)
        self.frozen_windows = _validated_windows(data, windows)
        self.identity = dict(kind='unit_window_v1', seed=seed,
            windows_sha256=hashlib.sha256(self.frozen_windows.astype('<i8').tobytes()).hexdigest(),
            data_sha256=hashlib.sha256(np.asarray(data.ids, dtype='<i8').tobytes()).hexdigest())
        if state is not None:
            _check_state_schema(state, {'cursor', 'pending', 'total_targets', 'record_targets',
                                       *self.identity}, self.identity)
        super().__init__(data, seed, state)
        _check_progress(self, self.windows[:, 1] - self.windows[:, 0] - 1, self.windows[:, 2])

    def _set_epoch(self, epoch):
        self.windows = self.frozen_windows
        self.cursor = BatchCursor(len(self.windows), 1, self.seed, epoch=epoch)

    def state(self):
        return {**super().state(), **self.identity}

    def next(self, quota, batch_size):
        _positive(quota, 'quota')
        _positive(batch_size, 'batch_size')
        return super().next(quota, batch_size)


class PoetryAuxStream:
    """Integer cumulative auxiliary quotas; substreams are already restored.

    Adapter construction restores state['domains'][name] into each substream
    first, then passes those objects and this complete mixture state here.
    """
    def __init__(self, main, auxiliary=None, numerator=1, denominator=5, state=None):
        _positive(numerator, 'numerator', allow_zero=True)
        _positive(denominator, 'denominator')
        if numerator >= denominator or (auxiliary is None) != (numerator == 0):
            raise ValueError('Invalid auxiliary fraction or active domains')
        self.numerator, self.denominator = numerator, denominator
        self.streams = {'poetry': main}
        if auxiliary is not None:
            self.streams['auxiliary'] = auxiliary
        self.total_targets = 0
        if state is not None:
            if (not isinstance(state, dict) or set(state) != {'total_targets', 'auxiliary_fraction', 'domains'}
                    or state['auxiliary_fraction'] != [numerator, denominator]
                    or not isinstance(state['domains'], dict) or set(state['domains']) != set(self.streams)):
                raise ValueError('Resume auxiliary mixture/domains changed')
            _positive(state['total_targets'], 'total_targets', allow_zero=True)
            if any(stream.state() != state['domains'][name] for name, stream in self.streams.items()):
                raise ValueError('Substreams must be restored to their saved mixture state')
            self.total_targets = state['total_targets']
        self._check_counts()

    def _check_counts(self):
        auxiliary = self.streams['auxiliary'].total_targets if 'auxiliary' in self.streams else 0
        if (auxiliary != self.total_targets * self.numerator // self.denominator
                or sum(stream.total_targets for stream in self.streams.values()) != self.total_targets):
            raise ValueError('Mixture counts disagree with cumulative integer quotas')

    def state(self):
        return dict(total_targets=self.total_targets,
                    auxiliary_fraction=[self.numerator, self.denominator],
                    domains={name: stream.state() for name, stream in self.streams.items()})

    def statistics(self):
        domains = {}
        for name in ('poetry', 'auxiliary'):
            if name in self.streams:
                stats = self.streams[name].statistics()
                domains[name] = dict(**stats, active=True,
                                     exposures=stats['targets'] / stats['targets_per_epoch'])
            else:
                domains[name] = dict(targets=0, exposures=0., active=False, targets_per_epoch=None)
        return dict(total_targets=self.total_targets,
                    auxiliary_fraction=[self.numerator, self.denominator], domains=domains)

    def next(self, quota, batch_size):
        _positive(quota, 'quota')
        _positive(batch_size, 'batch_size')
        before = self.total_targets * self.numerator // self.denominator
        auxiliary = (self.total_targets + quota) * self.numerator // self.denominator - before
        counts = {'poetry': quota - auxiliary, 'auxiliary': auxiliary}
        batches = []
        for name, count in counts.items():
            if count:
                batches.extend(self.streams[name].next(count, batch_size))
        if sum(int(labels[:, 1:].ne(-100).sum()) for _, labels in batches) != quota:
            raise AssertionError('Supervised targets differ from mixture quota')
        self.total_targets += quota
        self._check_counts()
        return batches, counts
