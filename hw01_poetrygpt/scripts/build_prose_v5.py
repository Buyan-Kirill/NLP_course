"""Prepare independently bounded historical prose; retain frozen v3 rows verbatim.

No model training. Sparse 16-word fingerprints identify duplicate editions/families;
full 12/20-word held-out indexes exclude quotation-bearing paragraphs. Exclusions
create fragment boundaries rather than joining text across removed paragraphs.
"""
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from tokenizers import Tokenizer
from build_prose_v3 import FragmentIndex, rolling, words, read_lines, write_json, write_lines

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit/prose_v5'
OUT = ROOT / 'data/prose_v5'
MANUAL_EXCLUSIONS = {
    'http://az.lib.ru/t/turgenew_i_s/text_1250.shtml': 'Root QC: unfinished narrative, explicit Не закончено marker.',
    'http://az.lib.ru/t/turgenew_i_s/text_1871_novaya_povest.shtml': 'Root QC: unfinished draft, abbreviated character names and chapter plan.',
    'http://az.lib.ru/t/turgenew_i_s/text_1871_silaev.shtml': 'Root QC: unfinished narrative in draft collection, explicit Не закончено marker.',
    'http://az.lib.ru/t/turgenew_i_s/text_0870.shtml': 'Root QC: plan of novel, not continuous finished prose.',
    'http://az.lib.ru/g/gercen_a_i/text_0430.shtml': 'Root QC: philosophical/publicistic collection with German-edition variants, outside selected narrative scope.',
    'http://az.lib.ru/k/karamzin_n_m/text_0840oldorfo.shtml':
        'Root sampled review: editorial introduction and essay about friendship; not narrative prose.',
    'http://az.lib.ru/l/lazhechnikow_i_i/text_0010.shtml':
        'Independent review: annotation and contents page, not novel body.',
    'http://az.lib.ru/o/odoewskij_w_f/text_0530.shtml':
        'Independent review: publicistic essay with unresolved OCR line-break hyphenation.',
    'http://az.lib.ru/g/gercen_a_i/text_1833_den_byl_dyshy.shtml':
        'Independent review: editorial alternative readings in tail; excluded rather than reconstruct edition.',
    'http://az.lib.ru/t/turgenew_i_s/text_0083.shtml': 'Modern-year review: scholarly discussion of unrealized drafts rather than a finished narrative.',
    'http://az.lib.ru/t/turgenew_i_s/text_0530.shtml': 'Modern-year review: mixed correspondence/prefaces/editorial collection, not narrative prose.',
    'http://az.lib.ru/t/turgenew_i_s/text_1883_le_stepovik.shtml': 'Modern-year review: translated late fragment, translation provenance/style not established.',
    'http://az.lib.ru/t/turgenew_i_s/text_1883_une_fin-1.shtml': 'Modern-year review: translated late fragment, translation provenance/style not established.',
    'http://az.lib.ru/k/krestowskij_w_w/text_0130.shtml': 'Modern-year review: modern introductory essay inseparable by current reviewed boundaries; deferred.',
    'http://az.lib.ru/o/odoewskij_w_f/text_1845_lektzii_pufa.shtml': 'Modern-year review: culinary lectures rather than narrative, with extensive modern commentary.'
}

def sha(text):
    return hashlib.sha256(text.encode('utf8') if isinstance(text, str) else text).hexdigest()

def family_title(title):
    title = re.sub(r'\b(?:том|часть|книга)\s+[IVX\d]+.*$', '', title, flags=re.I)
    return ' '.join(words(title))

def canonical_author(name):
    name = name.split('. ')[0]
    if name.startswith('Бестужев-Марлинский '):
        return 'А. А. Бестужев'
    parts = name.split()
    if len(parts) == 3:
        return f'{parts[1][0]}. {parts[2][0]}. {parts[0]}'
    return name

def strip_html(source):
    source = re.sub(r'<!--.*?-->', '', source, flags=re.S)
    source = re.sub(r'<(?:script|style)\b.*?</(?:script|style)>', '', source, flags=re.I | re.S)
    source = re.sub(r'<sup\b.*?</sup>', '', source, flags=re.I | re.S)
    source = re.sub(r'<a\b[^>]*href=["\']?#[^>]*>\s*(?:\d+|\[\d+\]|\*)\s*</a>', '', source, flags=re.I | re.S)
    # Physical HTML line wrapping is not a paragraph boundary. PRE blocks retain
    # blank-line paragraphs; other boundaries come only from block/BR tags.
    source = re.sub(r'(<pre\b[^>]*>)(.*?)(</pre>)',
                    lambda m: m[1] + re.sub(r'\n\s*\n', '<p>', m[2]) + m[3],
                    source, flags=re.I | re.S)
    source = source.replace('\n', ' ')
    source = re.sub(r'<(?:dd|br|p|/p|/div|/h[1-6]|/pre|/li)\b[^>]*>', '\n', source, flags=re.I)
    source = re.sub(r'<[^>]+>', '', source)
    return html.unescape(source)

def clean(entry):
    data = (AUDIT / 'raw' / entry['filename']).read_bytes()
    assert sha(data) == entry['sha256']
    source = data.decode('cp1251').replace('\r', '')
    opening = re.search(r'<!--[^>]*Собственно произведение[^>]*-->', source)
    if not opening:
        return [], [{'reason': 'missing_main_work_marker'}]
    end = source.find('</noindex>', opening.end())
    if end < 0:
        return [], [{'reason': 'missing_main_work_end'}]
    body = source[opening.end():end]
    # Explicit author-source association; this is not a claim of scholarly completeness.
    raw = [re.sub(r'[ \t\u00a0]+', ' ', x).strip() for x in strip_html(body).splitlines()]
    raw = [x for x in raw if x]
    ledger, paragraphs = [], []
    opening_index = next((i for i, p in enumerate(raw) if len(p) >= 80 and len(words(p)) >= 8
                          and sum(ch.islower() for ch in p) / max(1, sum(ch.isalpha() for ch in p)) > .75
                          and not re.search(r'издани|собраніе|собрание сочинений|подготов|источник|электрон|OCR|http|copyright|типограф|издатель|вступительн|библ[иі]ограф|составлен|Сост\.|вступ\. ст\.|примеч\.|^\[|^\*', p, re.I)), None)
    if opening_index is None:
        return [], [{'reason': 'no_substantial_narrative_opening'}]
    ledger.append({'reason': 'frontmatter_excluded', 'paragraphs': opening_index,
                   'text_sha256': sha('\n'.join(raw[:opening_index]))})
    for i, p in enumerate(raw[opening_index:], opening_index):
        if (re.fullmatch(r'(?:ПРИМЕЧАНИЯ(?: АВТОРА)?|КОММЕНТАРИИ|ПРИМЕЧАНИЯ И КОММЕНТАРИИ|СТАТЬИ И КОММЕНТАРИИ|УСЛОВНЫЕ СОКРАЩЕНИЯ|ПОСЛЕСЛОВИЕ|ОТ РЕДАКЦИИ|Примечания\s*\*?)[.:\s]*', p, re.I)
                or re.match(r'^(?:Оригинал|Исходник)\s+(?:текста\s+)?(?:здесь|тут)\s*:', p, re.I)
                or re.match(r'^Источник(?:\s+текста)?\s*:', p, re.I)
                or re.match(r'^Впервые\s*(?:[—-]|напечатан|опубликован)', p, re.I)):
            ledger.append({'reason': 'editorial_tail_excluded', 'paragraph_from': i,
                           'text_sha256': sha('\n'.join(raw[i:]))})
            break
        before = p
        p = unicodedata.normalize('NFC', p)
        # Deterministic spelling conversion, not lexical modernization or reconstruction.
        p = p.translate(str.maketrans('ѢѣІіѲѳѴѵ', 'ЕеИиФфИи'))
        p = re.sub(r'(?<=[А-Яа-яЁё])[ъЪ]\b', '', p)
        p = re.sub(r'\{[^{}]*\}|\[\d+\]', '', p)
        p = re.sub(r'(?<!-)--(?!-)', '—', p)
        p = re.sub(r'[ \t\u00a0]+', ' ', p).strip()
        reasons = []
        if re.search(r'https?://|www\.|ISBN|OCR|©|электронн(?:ая|ой) библиотек|подготовка текста|сканирован|оцифров|публикуется по|печат[а-я]+ по|собрание сочинений|РГАЛИ|ЦГАЛИ|ИРЛИ|издательство|типография|^Источник\s*:|^Цена за .*том|вступительной статьей|^\*.*\((?:фр|нем|лат|англ)\.\)|^\d+\s*\.\s*|^\[|\]$|Сост\.|вступ\. ст\.|примеч\.', p, re.I):
            reasons.append('editorial_or_source_metadata')
        if re.search(r'[А-Яа-яЁё][0-9]|[0-9][А-Яа-яЁё]|[А-Яа-яЁё][A-Za-z]|[A-Za-z][А-Яа-яЁё]|\ufffd|[А-Яа-яЁё][_`|][А-Яа-яЁё]', p):
            reasons.append('suspected_ocr_or_attached_note')
        if re.search(r'\[[^\]]+\]|\{[^}]+\}', p):
            reasons.append('unresolved_editorial_bracket')
        cyr = len(re.findall('[А-Яа-яЁё]', p)); latin = len(re.findall('[A-Za-z]', p))
        if latin > max(20, cyr * .15):
            reasons.append('foreign_language_paragraph')
        if not cyr:
            # A roman chapter number is a boundary, not a new document.
            continue
        if re.fullmatch(r'(?:1[789]\d\d)(?:\s*[—-]\s*1[789]\d\d)?\s*(?:г\.?|года)?\.?', p):
            ledger.append({'reason': 'standalone_year_kept_in_source_not_training_text', 'paragraph': i, 'text': p})
            continue
        if len(p) < 250 and re.search(r'Русская [Мм]ысль|Отечественн[ыяе]+ [Зз]аписки|Русский [Вв]естник', p) and re.search(r'\b(?:кн\.|No\s*\d|№|1[789]\d\d)', p):
            reasons.append('publication_footer')
        if before != p:
            ledger.append({'reason': 'orthography_dash_or_note_marker_normalization', 'paragraph': i,
                           'before_sha256': sha(before), 'after_sha256': sha(p)})
        if reasons:
            ledger.append({'reason': reasons, 'paragraph': i, 'text': before})
            paragraphs.append(None)
        else:
            paragraphs.append(p)
    return paragraphs, ledger

def fingerprint(text):
    return {h for _, h in rolling(words(text), 16) if h % 32 == 0}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((AUDIT / 'download_manifest.json').read_text(encoding='utf8'))
    old = read_lines(ROOT / 'data/prose_v3/corpus.jsonl')
    old_titles = {family_title(r['title'].split(':')[-1]) for r in old}
    old_titles |= {'тарaс бульба', 'тарас бульба', 'мертвые души', 'война и мир', 'братья карамазовы',
                   'бесы', 'подросток', 'униженные и оскорбленные', 'рославлев', 'соборяне'}
    index = defaultdict(set)
    for r in old:
        for h in fingerprint(r['text']):
            index[h].add(r['family_id'])
    print('Old fingerprints', len(index), flush=True)
    candidates, admissions, transforms = [], [], []
    for n, entry in enumerate(manifest['files']):
        if entry['url'] in MANUAL_EXCLUSIONS:
            admissions.append({'url': entry['url'], 'title': entry['title'], 'status': 'excluded',
                               'reason': 'reviewed_non_narrative_work', 'evidence': MANUAL_EXCLUSIONS[entry['url']]})
            continue
        if re.search(r'предисловие|послесловие|заметки о|записная книжка|статьи|литературн.*воспомин|биограф|письм[ао] к|обзор', entry['title'], re.I):
            admissions.append({'url': entry['url'], 'title': entry['title'], 'status': 'excluded', 'reason': 'editorial_or_essay_title'})
            continue
        title_key = family_title(entry['title'])
        if any(title_key == t or (len(t) >= 10 and title_key.startswith(t + ' ')) for t in old_titles):
            admissions.append({'url': entry['url'], 'title': entry['title'], 'status': 'excluded', 'reason': 'existing_v3_work_title_family'})
            continue
        paragraphs, ledger = clean(entry)
        source_head = (AUDIT / 'raw' / entry['filename']).read_bytes().decode('cp1251')[:10000]
        source_form = re.search(r'<a\s+href=["\']?/type/[^>]*>(.*?)</a>', source_head, re.I | re.S)
        entry['source_form'] = re.sub(r'<[^>]+>', '', source_form[1]).strip() if source_form else None
        transforms.append({'url': entry['url'], 'operations': ledger})
        text = '\n'.join(p or '' for p in paragraphs)
        fp = fingerprint(text)
        counts = Counter(f for h in fp for f in index.get(h, ()))
        top = counts.most_common(1)
        if top and top[0][1] >= 3 and (top[0][1] / max(1, len(fp)) >= .03 or top[0][1] >= 50):
            admissions.append({'url': entry['url'], 'title': entry['title'], 'status': 'excluded',
                               'reason': 'duplicate_or_compilation_overlap', 'matching_family': top[0][0],
                               'matching_fingerprints': top[0][1], 'fingerprints': len(fp)})
            continue
        if len(words(text)) < 80:
            admissions.append({'url': entry['url'], 'title': entry['title'], 'status': 'excluded', 'reason': 'too_little_clean_text'})
            continue
        family = 'az_family_' + sha(entry['author_path'] + ':' + title_key)[:20]
        for h in fp:
            index[h].add(family)
        split = 'validation' if int(sha('prose_v5_validation:' + family)[:8], 16) % 100 < 5 else 'train'
        candidates.append({'entry': entry, 'paragraphs': paragraphs, 'family': family, 'split': split})
        if n % 50 == 0:
            print('Cleaned', n, '/', len(manifest['files']), 'accepted', len(candidates), flush=True)
    del index
    write_json(AUDIT / 'candidate_summary.json', {'works': len(candidates), 'words': sum(len(words('\n'.join(p or '' for p in c['paragraphs']))) for c in candidates)})
    # Old frozen test/validation + newly assigned complete families. No old row changes.
    heldout = FragmentIndex(20)
    for r in old:
        if r['split'] != 'train':
            heldout.add(r['work_id'], r['text'])
    for c in candidates:
        if c['split'] == 'validation':
            heldout.add(c['family'], '\n'.join(p or '' for p in c['paragraphs']))
    poetry = FragmentIndex(12)
    poetry_lines = set()
    poetry_splits = {r['work_id']: r['split'] for r in read_lines(ROOT / 'data/poetry_v3/metadata.jsonl')}
    for r in read_lines(ROOT / 'data/poetry_v3/corpus.jsonl'):
        if poetry_splits[r['work_id']] != 'train':
            poetry.add(r['work_id'], r['text'])
            for line in r['text'].splitlines():
                w = words(line)
                if len(w) >= 4:
                    poetry_lines.add(' '.join(w))
    print('Heldout indexes', len(heldout.index), len(poetry.index), flush=True)
    tokenizer = Tokenizer.from_file(str(ROOT / 'artifacts/tokenizers_v3/selected/tokenizer.json'))
    rows, samples, exact, token_counts = list(old), [], set(), Counter()
    exclusions = []
    for c in candidates:
        kept, dropped = [], Counter()
        for i, p in enumerate(c['paragraphs']):
            reason = None
            if p is not None:
                if next(poetry.matches(p), None) or ' '.join(words(p)) in poetry_lines:
                    reason = 'heldout_poetry_quote'
                elif c['split'] == 'train' and next(heldout.matches(p), None):
                    reason = 'heldout_prose_20word_quote'
            if reason:
                exclusions.append({'family': c['family'], 'paragraph': i, 'reason': reason, 'text_sha256': sha(p)})
                dropped[reason] += 1
                kept.append(None)
            else:
                kept.append(p)
        chunks, chunk, start = [], [], 0
        for i, p in enumerate(kept + [None]):
            if p is None:
                if chunk:
                    chunks.append((start, i - 1, '\n\n'.join(chunk)))
                chunk = []
                start = i + 1
            else:
                chunk.append(p)
        entry = c['entry']; added = 0; total_tokens = 0
        for position, (first, last, text) in enumerate(chunks):
            if len(words(text)) < 80:
                continue
            normalized_hash = sha(' '.join(words(text)))
            if normalized_hash in exact:
                continue
            exact.add(normalized_hash)
            work_id = 'prose_az_' + sha(entry['url'])[:20] + '_s' + str(position).zfill(4)
            count = len(tokenizer.encode(text, add_special_tokens=False).ids)
            total_tokens += count
            row = {'work_id': work_id, 'original_work_id': c['family'], 'family_id': c['family'],
                   'split': c['split'], 'domain': 'prose', 'author': canonical_author(entry['author']), 'title': entry['title'],
                   'genre': {'Роман': 'novel', 'Повесть': 'novella', 'Рассказ': 'short_story', 'Очерк': 'sketch',
                             'Сборник': 'collection', 'Сказка': 'fairy_tale', 'Миниатюра': 'miniature'}.get(entry['source_form'], 'prose_narrative'),
                   'source_form': entry['source_form'], 'source_genre': entry['catalog_entry'], 'themes': [],
                   'written_from': None, 'written_to': None,
                   'date_basis': 'library_catalog_year_unverified_creation_vs_publication' if entry['source_year'] else 'selected_historical_author_unknown_work_year',
                   'source_date_values': [entry['source_year']] if entry['source_year'] else [],
                   'annotation_status': {'themes': 'unannotated', 'genre': 'source_catalog_heuristic'},
                   'sources': [{'dataset': 'Lib.ru_Classic', 'record_id': entry['filename'], 'url': entry['url'],
                                'retrieved_date': '2026-10-31', 'raw_sha256': entry['sha256'], 'catalog_url': entry['catalog_url']}],
                   'quality': {'is_excerpt': True, 'full_literary_proofreading': False, 'paragraph_from': first,
                               'paragraph_to': last, 'historical_spelling_conversion': 'yat_i_fita_izhitsa_and_final_hard_sign',
                               'embedded_verse_or_authorial_epigraphs_possible': True},
                   'token_count': count, 'text_sha256': sha(text), 'text': text}
            rows.append(row); added += 1; token_counts[c['split']] += count
        admissions.append({'url': entry['url'], 'title': entry['title'], 'family_id': c['family'], 'split': c['split'],
                           'status': 'accepted' if added else 'excluded', 'segments': added, 'tokens': total_tokens,
                           'quote_exclusions': dict(dropped)})
        source_rows = [r for r in rows[-added:] if added and r['family_id'] == c['family']]
        if source_rows:
            longest = max(source_rows, key=lambda r: len(r['text']))
            paras = longest['text'].split('\n\n')
            for fraction in (0, .5, .95):
                p = paras[min(len(paras)-1, int(len(paras)*fraction))]
                samples.append({'family_id': c['family'], 'author': longest['author'], 'title': entry['title'],
                                'fraction': fraction, 'text': p[:1400]})
        if len(admissions) % 50 == 0:
            print('Tokenized', len(admissions), 'new tokens', dict(token_counts), flush=True)
    write_lines(OUT / 'corpus.jsonl', rows)
    write_lines(OUT / 'metadata.jsonl', [{k: v for k, v in r.items() if k != 'text'} for r in rows])
    for split in ('train', 'validation', 'test'):
        write_lines(OUT / (split + '.jsonl'), [{'work_id': r['work_id'], 'text': r['text']} for r in rows if r['split'] == split])
    write_json(OUT / 'split_assignments.json', {r['work_id']: {'split': r['split'], 'family_id': r['family_id']} for r in rows})
    write_json(AUDIT / 'admissions.json', admissions)
    write_lines(AUDIT / 'transformations.jsonl', transforms)
    write_lines(AUDIT / 'quote_exclusions.jsonl', exclusions)
    write_lines(AUDIT / 'editorial_sample.jsonl', samples)
    summary = {'release': 'prose_v5', 'new_token_counts': dict(token_counts), 'old_rows_unchanged': len(old),
               'splits': {s: {'segments': sum(r['split'] == s for r in rows),
                              'families': len({r['family_id'] for r in rows if r['split'] == s}),
                              'authors': len({r['author'] for r in rows if r['split'] == s})}
                          for s in ('train', 'validation', 'test')},
               'rejections': dict(Counter(a.get('reason', 'accepted') for a in admissions)),
               'limits': ['Source catalog dates may be publication, creation, or erroneous; unknown dates retained.',
                          'Automated cleaning and sampled review are not full scholarly proofreading.',
                          'HTTP public source offers no TLS authentication; stored raw hashes pin downloaded evidence.',
                          'Some sources are multi-story collections, treated as a single family, never split across partitions.',
                          'Deduplication uses sparse exact fingerprints and conservative full-work exclusions; all near duplicates not guaranteed absent.'],
               'manifest_sha256': sha((AUDIT/'download_manifest.json').read_bytes()),
               'builder_sha256': sha(Path(__file__).read_bytes())}
    write_json(AUDIT / 'report.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
