"""Build source-faithful prose excerpts, with explicit exclusions and family splits."""
import hashlib
import json
import re
import unicodedata
import zlib
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit/prose_v3'
OUT = ROOT / 'data/prose_v3'
WORD = re.compile(r'[а-яё]+', re.I)
MASK = (1 << 64) - 1
BASE = 1000003


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode('utf-8')).hexdigest()


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def words(text):
    return [m.group().lower().replace('ё', 'е') for m in WORD.finditer(text)]


def rolling(sequence, n):
    codes = [zlib.crc32(word.encode()) + 1 for word in sequence]
    if len(codes) < n:
        return
    factor = pow(BASE, n - 1, 1 << 64)
    value = 0
    for code in codes[:n]:
        value = (value * BASE + code) & MASK
    yield 0, value
    for i in range(n, len(codes)):
        value = ((value - codes[i - n] * factor) * BASE + codes[i]) & MASK
        yield i - n + 1, value


class FragmentIndex:
    def __init__(self, n):
        self.n, self.index, self.documents = n, {}, {}

    def add(self, identifier, text):
        sequence = words(text)
        self.documents[identifier] = sequence
        for pos, value in rolling(sequence, self.n):
            self.index.setdefault(value, (identifier, pos))

    def matches(self, text):
        tokens = list(WORD.finditer(text))
        sequence = [m.group().lower().replace('ё', 'е') for m in tokens]
        for pos, value in rolling(sequence, self.n):
            other = self.index.get(value)
            if other:
                identifier, offset = other
                if sequence[pos:pos + self.n] == self.documents[identifier][offset:offset + self.n]:
                    yield tokens[pos].start(), tokens[pos + self.n - 1].end(), identifier


def family_key(filename):
    stem = Path(filename).stem
    return re.sub(r'(?:_b\d+|(?<=karenina)\d+|(?<=vojna)\d+)$', '', stem)


def split_for(family):
    number = int(sha('prose_v3_fixed_split_20261031:' + family)[:8], 16) % 100
    return 'train' if number < 80 else 'validation' if number < 90 else 'test'


def clean(entry):
    path = AUDIT / 'raw' / entry['filename']
    data = path.read_bytes()
    expected = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    assert expected == entry['blob_sha1'] and len(data) == entry['bytes'], path
    text = unicodedata.normalize('NFC', data.decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n'))
    lines = text.splitlines()
    if entry['filename'] == 'dostoevsky_prestuplenie.txt':
        assert sha('\n'.join(lines[355:418])) == '2aa5b02c73d70b49f7a08a0da8c3881ae91e605ef6da1a380734d58de9a06a39'
    ledger, kept = [], []
    # Remove only initial front matter; excerpts are explicitly not claimed complete editions.
    start = next((i for i, line in enumerate(lines) if len(line.strip()) >= 180 and len(words(line)) >= 24), None)
    if start is None:
        return [], [{'reason': 'no_substantial_prose_opening'}]
    if start:
        ledger.append({'reason': 'opening_frontmatter_and_epigraph_excluded', 'line_from': 1, 'line_to': start,
                       'text_sha256': sha('\n'.join(lines[:start]))})
    for i, line in enumerate(lines[start:], start):
        stripped = line.strip()
        if entry['filename'] == 'dostoevsky_prestuplenie.txt' and 356 <= i + 1 <= 418:
            ledger.append({'reason': 'reviewed_duplicate_chapter_excluded', 'source_line': i + 1, 'text': line,
                           'evidence': 'audit/prose_v3/within_train_suspect_dostoevsky.json'})
            kept.append({'text': None, 'source_line': i + 1})
            continue
        if entry['filename'] == 'bestuzhev_mulla.txt' and 351 <= i + 1 <= 356:
            ledger.append({'reason': 'reviewed_complete_heldout_poetry_quote_excluded',
                           'source_line': i + 1, 'text': line,
                           'matches': ['work_eeab88a60d73a37d']})
            kept.append({'text': None, 'source_line': i + 1})
            continue
        if i > len(lines) // 2 and (re.fullmatch(r'(?:ПРИМЕЧАНИЯ|КОММЕНТАРИИ|СНОСКИ)', stripped, re.I)
                                    or re.match(r'^\[1\]\s+\S', stripped)):
            ledger.append({'reason': 'trailing_editorial_notes_excluded', 'line_from': i + 1, 'line_to': len(lines),
                           'text_sha256': sha('\n'.join(lines[i:]))})
            break
        if not stripped:
            kept.append({'text': '', 'source_line': i + 1})
            continue
        reasons = []
        revised = re.sub(r'_([^_\n]+)_', r'\1', stripped)
        revised = re.sub(r'(?<=[А-Яа-яЁё]),(?=[А-Яа-яЁё])', ', ', revised)
        revised = re.sub(r'(?<=[а-яё])([.!?;:])(?=[А-ЯЁ])', r'\1 ', revised)
        revised = re.sub(r'([А-Яа-яЁё][,.])(?=\d{1,2}-го (?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря))',
                         r'\1 ', revised)
        if revised != stripped:
            ledger.append({'reason': 'source_emphasis_and_missing_punctuation_spaces_normalized',
                           'source_line': i + 1, 'before': stripped, 'after': revised})
            stripped = revised
        if re.search(r'[А-Яа-яЁё][,.;:!?][0-9]{1,3}(?:\s|$)', stripped):
            reasons.append('punctuation_attached_note_number_requires_review')
        if re.search(r'[А-Яа-яЁё][0-9]|[0-9][А-Яа-яЁё]|[А-Яа-яЁё][_^`|][А-Яа-яЁё]|\ufffd', stripped):
            reasons.append('suspected_ocr_or_attached_footnote')
        if re.search(r'https?://|</?[A-Za-z][^>]*>|^\s*(?:Редакторы|ISBN|OCR)\s*:', stripped, re.I):
            reasons.append('editorial_or_markup')
        if re.search(r'\[\d+\]|\{\d+\}', stripped):
            revised = re.sub(r'\[\d+\]|\{\d+\}', '', stripped)
            ledger.append({'reason': 'numeric_bracket_note_markers_removed', 'source_line': i + 1,
                           'before': stripped, 'after': revised})
            stripped = revised
        if re.search(r'\[[^\]]{2,}\]|\{[^}]{2,}\}', stripped):
            reasons.append('inline_editorial_or_translation_note_requires_review')
        # Standalone short chapter labels/separators carry little language content.
        if not re.search(r'[А-Яа-яЁё]', stripped):
            reasons.append('non_russian_line_or_separator')
        if reasons:
            ledger.append({'reason': reasons, 'source_line': i + 1, 'text': stripped})
            kept.append({'text': None, 'source_line': i + 1})
        else:
            kept.append({'text': re.sub(r'[ \t\u00a0]+', ' ', stripped), 'source_line': i + 1})
    return kept, ledger


def mark_fragments(lines, index, reason, ledger):
    text = '\n'.join(line['text'] or '' for line in lines)
    offsets, position = [], 0
    for line in lines:
        offsets.append(position)
        position += len(line['text'] or '') + 1
    hits = defaultdict(set)
    for first, last, identifier in index.matches(text):
        for i in range(bisect_right(offsets, first) - 1, bisect_right(offsets, last - 1)):
            hits[i].add(identifier)
    for i in sorted(hits):
        if lines[i]['text']:
            ledger.append({'reason': reason, 'source_line': lines[i]['source_line'], 'text': lines[i]['text'],
                           'matches': sorted(hits[i])})
            lines[i]['text'] = None
    return len(hits)


def segments(lines):
    segment = []
    for line in lines + [{'text': None}]:
        if line['text'] is None:
            if segment:
                yield segment
            segment = []
        else:
            segment.append(line)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((AUDIT / 'download_manifest.json').read_text(encoding='utf-8'))
    poetry_file = ROOT / 'data/poetry_v3/corpus.jsonl'
    assert poetry_file.exists(), 'Finalize poetry_v3 before preparing prose'
    poetry = FragmentIndex(12)
    for row in read_lines(poetry_file):
        poetry.add(row['work_id'], row['text'])
    heldout_lines = {}
    for split in ('validation', 'test'):
        for row in read_lines(ROOT / 'data/poetry_v3' / (split + '.jsonl')):
            for line in row['text'].splitlines():
                sequence = words(line)
                if len(sequence) >= 4:
                    heldout_lines.setdefault(' '.join(sequence), row['work_id'])
    print('Poetry fragment index', len(poetry.index), flush=True)
    cleaned, ledgers, admissions = [], [], []
    for entry in manifest['files']:
        lines, ledger = clean(entry)
        for line in lines:
            normalized = ' '.join(words(line['text'] or ''))
            if normalized in heldout_lines and normalized != 'слава богу слава богу':
                ledger.append({'reason': 'exact_heldout_poetry_line_4plus_words_excluded',
                               'source_line': line['source_line'], 'text': line['text'],
                               'matches': [heldout_lines[normalized]]})
                line['text'] = None
        mark_fragments(lines, poetry, 'poetry_fragment_12_words_excluded', ledger)
        family = family_key(entry['filename'])
        cleaned.append((entry, family, split_for(family), lines))
        ledgers.append({'filename': entry['filename'], 'operations': ledger})
    del poetry
    # Remove cross-partition 20-word repetitions, preserving all work parts in one split.
    test_index, heldout_index = FragmentIndex(20), FragmentIndex(20)
    for entry, family, split, lines in cleaned:
        if split == 'test':
            test_index.add(entry['filename'], '\n'.join(x['text'] or '' for x in lines))
    for i, (entry, family, split, lines) in enumerate(cleaned):
        if split == 'validation':
            mark_fragments(lines, test_index, 'prose_test_fragment_20_words_excluded', ledgers[i]['operations'])
    for entry, family, split, lines in cleaned:
        if split != 'train':
            heldout_index.add(entry['filename'], '\n'.join(x['text'] or '' for x in lines))
    for i, (entry, family, split, lines) in enumerate(cleaned):
        if split == 'train':
            mark_fragments(lines, heldout_index, 'prose_heldout_fragment_20_words_excluded', ledgers[i]['operations'])
    rows, metadata, exact, samples = [], [], set(), []
    for entry, family, split, lines in cleaned:
        meta = entry['metadata'][0]
        accepted_words, segment_count, discarded_words = 0, 0, 0
        for position, segment in enumerate(segments(lines)):
            text = re.sub(r'\n{3,}', '\n\n', '\n'.join(x['text'] for x in segment)).strip()
            count = len(words(text))
            if count < 64:
                discarded_words += count
                continue
            norm_hash = sha(' '.join(words(text)))
            if norm_hash in exact:
                continue
            exact.add(norm_hash)
            identifier = 'prose_' + Path(entry['filename']).stem + '_s' + str(position).zfill(4)
            record = {
                'work_id': identifier, 'original_work_id': family, 'family_id': 'prose_family_' + family,
                'split': split, 'domain': 'prose', 'author': meta['author'], 'title': meta['title'],
                'genre': 'prose_narrative', 'source_genre': meta['genre'], 'themes': [],
                'written_from': None, 'written_to': None, 'date_basis': 'source_curated_19th_century_unverified_work_date',
                'source_date_values': sorted({r['year'] for r in entry['metadata']}),
                'source_metadata': entry['metadata'], 'annotation_status': {'themes': 'unannotated', 'genre': 'source_derived'},
                'sources': [{'dataset': 'RSD', 'record_id': entry['path'], 'url': entry['url'],
                             'revision': manifest['commit'], 'edition': meta['source']}],
                'quality': {'is_excerpt': True, 'source_line_from': segment[0]['source_line'],
                            'source_line_to': segment[-1]['source_line'], 'full_literary_proofreading': False,
                            'embedded_verse_or_authorial_epigraphs_possible': True},
                'text_sha256': sha(text), 'text': text,
            }
            rows.append(record)
            metadata.append({k: v for k, v in record.items() if k != 'text'})
            accepted_words += count
            segment_count += 1
        admissions.append({'filename': entry['filename'], 'family': family, 'split': split, 'accepted_segments': segment_count,
                           'accepted_words': accepted_words, 'short_segment_words_excluded': discarded_words,
                           'metadata_conflict': len({r['year'] for r in entry['metadata']}) > 1})
        # Deterministic coverage sample for subsequent editorial reading, not a measured error-rate sample.
        source_rows = [r for r in rows if r['sources'][0]['record_id'] == entry['path']]
        if source_rows:
            longest = max(source_rows, key=lambda r: len(r['text']))
            paras = [s for s in longest['text'].splitlines() if len(s) > 100]
            for fraction in (0, .5, .95):
                if paras:
                    text = paras[min(len(paras)-1, int(len(paras)*fraction))]
                    samples.append({'filename': entry['filename'], 'work_id': longest['work_id'],
                                    'position_fraction': fraction, 'text': text[:1600]})
    write_lines(OUT / 'corpus.jsonl', rows)
    write_lines(OUT / 'metadata.jsonl', metadata)
    assignments = {r['work_id']: {'split': r['split'], 'family_id': r['family_id']} for r in rows}
    write_json(OUT / 'split_assignments.json', assignments)
    for split in ('train', 'validation', 'test'):
        write_lines(OUT / (split + '.jsonl'), [{'work_id': r['work_id'], 'text': r['text']} for r in rows if r['split'] == split])
    write_lines(AUDIT / 'transformations.jsonl', ledgers)
    write_json(AUDIT / 'admissions.json', admissions)
    write_lines(AUDIT / 'editorial_sample.jsonl', samples)
    by_split = {}
    for split in ('train', 'validation', 'test'):
        group = [r for r in rows if r['split'] == split]
        by_split[split] = {'segments': len(group), 'families': len({r['family_id'] for r in group}),
                           'authors': len({r['author'] for r in group}), 'words': sum(len(words(r['text'])) for r in group),
                           'utf8_bytes': sum(len(r['text'].encode()) for r in group)}
    report = {'release': 'prose_v3', 'input_manifest_sha256': sha((AUDIT/'download_manifest.json').read_bytes()),
              'poetry_corpus_sha256': sha(poetry_file.read_bytes()), 'builder_sha256': sha(Path(__file__).read_bytes()),
              'input_files': len(manifest['files']), 'families': len({r['family_id'] for r in rows}),
              'splits': by_split, 'authors_segment_counts': dict(Counter(r['author'] for r in rows)),
              'selection': 'Pinned RSD first/second half XIX prose collections; brevia duplicates omitted.',
              'license': 'Repository GPL-3.0 text preserved; underlying editions and attribution retained; no external redistribution performed.',
              'limits': ['Prose excerpts, not complete scholarly editions; removals and boundaries are explicit.',
                         'Source dates may mix creation/publication and contain mistakes; not independently verified.',
                         'Single curated repository with multiple upstream editions; themes not yet annotated.',
                         'Heuristic OCR flags and quotation matching can remove valid text; complete literary proofreading not claimed.',
                         'Family split is deterministic by original title stem, not by segment; heldout sizes must be reported.',
                         'Newline conventions are retained from source; historical orthography is not silently rewritten.'],
              'output_sha256': {p.name: sha(p.read_bytes()) for p in sorted(OUT.glob('*.json*'))}}
    write_json(AUDIT / 'report.json', report)
    print(json.dumps(by_split, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
