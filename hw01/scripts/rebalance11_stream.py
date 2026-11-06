"""Exact-quota mixtures of the inherited prose stream and selected target spans.

Span coordinates are absolute offsets in the unchanged prepared token array.
Targets never overlap within the selected pool. Like EpochStream, each window
starts with its real preceding token; partial-quota resumes retain that window
prefix without adding any extra context. No token or EOS is synthesized.
"""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from poetrygpt.data import BatchCursor
from scripts.train_causal_series import padded
from scripts.train_pretrain08 import EpochStream


def read_spans(path):
    with Path(path).open(encoding='utf-8-sig') as source:
        return [json.loads(line) for line in source if line.strip()]


class SpanStream:
    def __init__(self, data, spans, seed, state=None):
        self.data, self.spans, self.seed = data, deepcopy(spans), seed
        if not self.spans or data.context_length < 2:
            raise ValueError('Nonempty spans and context >= 2 required')
        intervals = {}
        for span in self.spans:
            owner = span['source_record_index']
            if type(owner) is not int or not 0 <= owner < len(data.metadata):
                raise ValueError('Invalid span owner')
            record = data.metadata[owner]
            first, stop, context = (span[k] for k in ('target_start', 'target_end', 'context_start'))
            if (any(type(v) is not int for v in (first, stop, context))
                    or not record['start'] - 1 <= context < first < stop <= record['end']
                    or first < record['start']):
                raise ValueError('Span targets/context cross their source record')
            for key in ('work_id', 'family_id'):
                if span.get(key) != record.get(key):
                    raise ValueError('Span provenance differs from prepared record')
            eos = data.manifest['special_tokens']['<|eos|>']
            if eos in data.ids[context + 1:stop - 1]:
                raise ValueError('Span contains an interior EOS')
            intervals.setdefault(owner, []).append((first, stop))
        for values in intervals.values():
            ordered = sorted(values)
            if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
                raise ValueError('Overlapping supervised spans')
        self.targets_per_epoch = sum(s['target_end'] - s['target_start'] for s in spans)
        self.total_targets, self.pending = 0, None
        self.span_targets = np.zeros(len(spans), dtype=np.int64)
        self.record_targets = np.zeros(len(data.metadata), dtype=np.int64)
        self._set_epoch(0 if state is None else state['cursor']['epoch'])
        if state is not None:
            self.cursor = BatchCursor(len(self.windows), 1, seed, **state['cursor'])
            self.pending = deepcopy(state['pending'])
            self.total_targets = state['total_targets']
            self.span_targets = np.asarray(state['span_targets'], dtype=np.int64)
            self.record_targets = np.asarray(state['record_targets'], dtype=np.int64)
            if (self.span_targets.shape != (len(spans),)
                    or self.record_targets.shape != (len(data.metadata),)
                    or np.any(self.span_targets < 0) or np.any(self.record_targets < 0)
                    or self.span_targets.sum() != self.total_targets
                    or self.record_targets.sum() != self.total_targets):
                raise ValueError('Invalid selected-stream coverage state')
            if self.pending is not None:
                index, consumed = self.pending['index'], self.pending['consumed']
                if (not 0 <= index < len(self.windows)
                        or not 0 <= consumed < self.windows[index][1] - self.windows[index][0]):
                    raise ValueError('Invalid pending selected window')

    def _set_epoch(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        self.windows = []
        width = self.data.context_length - 1
        for index, span in enumerate(self.spans):
            first, end = span['target_start'], span['target_end']
            # Change target cuts each epoch without losing first/last positions.
            stop = min(end, first + int(rng.integers(1, width + 1)))
            while first < end:
                self.windows.append((first, stop, index))
                first, stop = stop, min(end, stop + width)
        self.cursor = BatchCursor(len(self.windows), 1, self.seed, epoch=epoch)

    def state(self):
        return dict(cursor=self.cursor.state_dict(), pending=deepcopy(self.pending),
                    total_targets=self.total_targets, span_targets=self.span_targets.tolist(),
                    record_targets=self.record_targets.tolist())

    def statistics(self):
        return dict(targets=self.total_targets, targets_per_epoch=self.targets_per_epoch,
                    epoch=self.cursor.epoch, epoch_offset=self.cursor.offset,
                    spans_seen=int(np.count_nonzero(self.span_targets)),
                    records_seen=int(np.count_nonzero(self.record_targets)))

    def next(self, quota, batch_size):
        if type(quota) is not int or quota < 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError('Positive integer quota and batch size required')
        records = []
        while quota:
            if self.pending is None:
                if self.cursor.offset == self.cursor.size:
                    self._set_epoch(self.cursor.epoch + 1)
                self.pending = dict(index=self.cursor.next()[0], consumed=0)
            begin, end, index = self.windows[self.pending['index']]
            first = begin + self.pending['consumed']
            stop = min(end, first + quota)
            span = self.spans[index]
            # Match EpochStream: one preceding token at a fresh window, and
            # retain already consumed targets as context on a partial resume.
            # context_start is provenance for review, not extra training input.
            input_start = begin - 1
            ids = np.asarray(self.data.ids[input_start:stop], dtype=np.int64).tolist()
            labels = [-100] * (first - input_start) + ids[first - input_start:]
            records.append((ids, labels))
            count = stop - first
            self.total_targets += count
            self.span_targets[index] += count
            self.record_targets[span['source_record_index']] += count
            self.pending['consumed'] += count
            quota -= count
            if stop == end:
                self.pending = None
        return [padded(records[i:i + batch_size], self.data.pad_id)
                for i in range(0, len(records), batch_size)]


class MixedStream:
    """Cumulative exact quota, independent of update sizes and Torch global RNG."""
    def __init__(self, ordinary, selected, fraction, state=None):
        self.ordinary, self.selected = ordinary, selected
        self.fraction = Fraction(str(fraction))
        if not 0 <= self.fraction <= 1 or (self.fraction and selected is None):
            raise ValueError('Invalid mixture fraction/selected stream')
        self.parent_targets = ordinary.total_targets if state is None else state['parent_targets']
        self.local_targets = 0 if state is None else state['local_targets']
        self.extra_targets = 0 if state is None else state['extra_targets']
        if (self.extra_targets != self.expected_extra(self.local_targets)
                or ordinary.total_targets != self.parent_targets + self.local_targets - self.extra_targets
                or (selected is not None and selected.total_targets != self.extra_targets)):
            raise ValueError('Mixture and constituent stream counters disagree')

    def expected_extra(self, total):
        return total * self.fraction.numerator // self.fraction.denominator

    @property
    def total_targets(self):
        return self.parent_targets + self.local_targets

    def state(self):
        return dict(parent_targets=self.parent_targets, local_targets=self.local_targets,
                    extra_targets=self.extra_targets, ordinary=self.ordinary.state(),
                    selected=self.selected.state() if self.selected is not None else None)

    def statistics(self):
        return dict(local_targets=self.local_targets, ordinary_local_targets=self.local_targets - self.extra_targets,
                    selected_targets=self.extra_targets, ordinary=self.ordinary.statistics(),
                    selected=self.selected.statistics() if self.selected is not None else None)

    def next(self, quota, batch_size):
        if type(quota) is not int or quota < 1:
            raise ValueError('Positive integer quota required')
        extra = self.expected_extra(self.local_targets + quota) - self.extra_targets
        batches = self.ordinary.next(quota - extra, batch_size) if quota > extra else []
        if extra:
            batches += self.selected.next(extra, batch_size)
        self.local_targets += quota
        self.extra_targets += extra
        return batches
