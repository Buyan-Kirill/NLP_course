"""Bounded classical verse candidate collection; never changes accepted corpora.

Uses the incrementally collected public catalogue inventory. Every output remains
unassigned and unaccepted pending individual work-boundary and cross-stage review.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit/poetry_v7_candidates'
OUT = ROOT / 'data/poetry_v7_candidates'
QC_ISSUES = {
    'Кто озарит нас правды светом?.': 'Source punctuation artifact ?. requires source corroboration.',
    '...Се стою при дверехъ и толку:': 'Biblical prose epigraph and obsolete/OCR-like forms remain before poem.',
    'Мраморный дворец\n\nА. А. ФЕТУ': 'Two poems merged across undated boundary; intermediate place and new heading remain.',
    'Мраморный дворец\n\nНА СМЕРТЬ ГРАФИНИ': 'Two poems merged across undated boundary; individual work dates unreliable.',
    'С лазоревой пролившись тишины.': 'Suspicious source reading: repeated tishiny rhyme and semantically awkward azure silence; verify, do not guess vyshiny.',
    'Зноем томящий сад': 'Suspicious ungrammatical source reading tomyashchiy sad; verify another edition, no speculative correction.',
    'СТРОФЫ\n\nСевастиан-Мученик': 'Dedication and narrative poem merged; epigraph/headings and obvious Poli OCR readings remain.',
    'Душа моя грустью обьята': 'Obvious source typo: soft sign instead of hard sign in obyata; quarantine pending corroboration.',
    'Ясным солнцем ровняет свой взор': 'Suspicious source reading rovnyaet svoj vzor; verify another edition, do not guess correction.',
    'Шепот. Робкое дыханье,': 'Short direct citation of a known Fet poem; exclude pending citation-level split-family review.',
}


def sha(value):
    return hashlib.sha256(value.encode('utf8') if isinstance(value, str) else value).hexdigest()


def norm(text):
    return ' '.join(re.findall('[а-яa-z0-9]+', (text or '').lower().replace('ё', 'е')))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()] if path.exists() else []


def write_rows(path, values):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in values), encoding='utf8')


def extract(raw):
    source = raw.decode('cp1251').replace('\r', '')
    match = re.search(r'<!--[^>]*Собственно произведение[^>]*-->', source)
    if not match or '</noindex>' not in source[match.end():]:
        return '', ['missing_explicit_work_container']
    body = source[match.end():source.index('</noindex>', match.end())]
    decisions = ['explicit_work_comment_to_noindex; dd/br/p and pre physical line breaks retained']
    body = re.sub(r'<!--.*?-->', '', body, flags=re.S)
    body = re.sub(r'<(?:script|style)\b.*?</(?:script|style)>', '', body, flags=re.I | re.S)
    # Headings are metadata, not verse. Record presence of multiple headings as a
    # collection-boundary warning rather than pretending the page is one poem.
    headings = re.findall(r'<h[1-6]\b.*?</h[1-6]>', body, flags=re.I | re.S)
    if len(headings) > 2:
        decisions.append('multiple_headings_require_individual_work_segmentation')
    body = re.sub(r'<h[1-6]\b.*?</h[1-6]>', '', body, flags=re.I | re.S)
    # Only pre-verse italic source credits are removed; later italic verse stays.
    first_i = re.search(r'<i\b[^>]*>.*?</i>', body, flags=re.I | re.S)
    if first_i and re.search(r'источник|типограф|OCR|собрание сочинений|электрон', first_i[0], re.I):
        body = body[:first_i.start()] + body[first_i.end():]
        decisions.append('leading_italic_source_credit_removed')
    protected = []
    def pre(m):
        protected.append(m[1]); return f'POETRYPRE{len(protected)-1}END'
    body = re.sub(r'<pre\b[^>]*>(.*?)</pre>', pre, body, flags=re.I | re.S)
    body = body.replace('\n', ' ')
    body = re.sub(r'<(?:dd|br|p|/p|/div|/li)\b[^>]*>', '\n', body, flags=re.I)
    for i, value in enumerate(protected):
        body = body.replace(f'POETRYPRE{i}END', value)
    body = html.unescape(re.sub(r'<[^>]+>', '', body))
    lines = [re.sub(r'[ \t\u00a0]+', ' ', x).strip() for x in body.splitlines()]
    text = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()
    return text, decisions


def individual_poems(parents):
    """Yield explicit mytag works and dated KR works, never scholarly prose."""
    for parent in parents:
        source = (ROOT / parent['snapshot']).read_bytes().decode('cp1251').replace('\r', '')
        if '/f/fofanow_k_m/text_0050.shtml' in parent['source_url']:
            matches = list(re.finditer(r'<mytag\s+var=["\']text["\']\s*>(.*?)</mytag>', source, re.S | re.I))
            for i, match in enumerate(matches):
                before = source[max(0, match.start() - 600):match.start()]
                after = source[match.end():matches[i+1].start() if i+1 < len(matches) else source.index('</noindex>', match.end())]
                names = re.findall(r'<mytag\s+var=["\']name["\']\s*>(.*?)</mytag>', before, re.S | re.I)
                years = re.findall(r'<mytag\s+var=["\']year["\']\s*>(.*?)</mytag>', after, re.S | re.I)
                year = re.search(r'\b(1[789]\d\d)\b', years[0]) if years else None
                if not year or not 1700 <= int(year[1]) <= 1900:
                    continue
                text = html.unescape(re.sub('<[^>]+>', '', re.sub(r'<br\s*/?>', '\n', match[1], flags=re.I))).strip()
                text = '\n'.join(re.sub(r'[ \t\u00a0]+', ' ', x).strip() for x in text.splitlines())
                title = re.sub(r'\s+', ' ', html.unescape(re.sub('<[^>]+>', '', names[-1]))).strip() if names else ''
                if not title or re.fullmatch(r'[ *]+', title):
                    title = next(x for x in text.splitlines() if x.strip())
                yield parent, title, text, int(year[1]), {'kind': 'explicit_mytag_text_name_year', 'source_date_literal': years[0], 'text_match_index': i, 'stanza_boundaries': 'explicit_br_br'}
        if '/k/kr_k_k/text_0040.shtml' in parent['source_url']:
            # First edition block only. Later bibliographical commentary and a
            # second edition follow this exact boundary; never treat them as verse.
            text = parent['text'].split('Поэты 1880-1890-х годов.')[0]
            text = text[text.index('Задремали волны,'):]
            lines = text.splitlines()
            previous = 0
            date_pattern = re.compile(r'^(?:(?:\d{1,2}\s+)?(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря|январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь|Осень|Страстная Среда)\s+)?(1[89]\d\d)$', re.I)
            for i, line in enumerate(lines):
                date = date_pattern.fullmatch(line.strip())
                if not date:
                    continue
                block = lines[previous:i]; previous = i + 1
                if int(date[1]) > 1900:
                    continue
                while block and not block[0].strip(): block.pop(0)
                while block and not block[-1].strip(): block.pop()
                removed_prefix, removed_tail = [], []
                title = ''
                while block:
                    s = block[0].strip()
                    structural = not s or s.isupper() or re.fullmatch(r'[ *IVXLCDM]+', s) or re.match(r'^(?:Из цикла|[А-Я]\.\s*[А-Я]\.|Барону|Графине|Князю|Княгине|Великому Кня|Великой Кня|Принцессе|Принцу|Королеве Эллинов|На помолвку|На пятидесятилетие|[Вв] ответ на его)', s)
                    structural = structural or (s.startswith('"') and s.endswith('"')) or s in ('Баркарола', 'Мост вздохов', 'Часовому', 'Пред увольнением', 'Новобранцу', 'Надпись к картине')
                    if structural:
                        if (s.isupper() and not re.fullmatch('[IVXLCDM]+', s)) or s in ('Баркарола', 'Мост вздохов', 'Часовому', 'Пред увольнением', 'Новобранцу'): title = s
                        removed_prefix.append(block.pop(0)); continue
                    break
                while block and (not block[-1].strip() or (not re.search(r'[.,!?:;…»"]$', block[-1].strip()) and len(block[-1].split()) <= 6)):
                    removed_tail.insert(0, block.pop())
                if not block:
                    continue
                poem = '\n'.join(block).strip()
                if not title: title = next(x for x in block if x.strip())
                yield parent, title, poem, int(date[1]), {'kind': 'first_KR_edition_only_dated_poem', 'source_date_literal': line, 'removed_prefix': removed_prefix, 'removed_location_tail': removed_tail, 'stanza_boundaries': 'pre_blank_lines', 'editorial_stop_literal': 'Поэты 1880-1890-х годов.'}


def main():
    if (ROOT / 'artifacts/release_v7_manifest.json').exists():
        raise ValueError('v7 is frozen; extend the collection in a new release')
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=85)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--existing-only', action='store_true')
    parser.add_argument('--top-up-danilevsky', action='store_true')
    parser.add_argument('--extra-author', action='append', default=[])
    parser.add_argument('--initialize-qc-after-full-reading', action='store_true')
    args = parser.parse_args()
    (AUDIT / 'raw').mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = rows(ROOT / 'audit/web_v7/inventory.jsonl')
    known, exact, lineidx, titleidx, knownurls = [], {}, defaultdict(set), defaultdict(set), set()
    assignments = {}
    for base in [ROOT / 'data/poetry_v6', ROOT / 'data/poetry_v6/reserve']:
        assignments.update(json.loads((base / 'split_assignments.json').read_text(encoding='utf8')))
        known.extend(rows(base / 'corpus.jsonl'))
    knownmap = {r['work_id']: r for r in known}
    for r in known:
        wid = r['work_id']; exact[norm(r['text'])] = wid
        titleidx[(norm(r.get('author')).split(' ')[0], norm(r.get('title')))].add(wid)
        for line in r['text'].splitlines():
            n = norm(line)
            if len(n) >= 18 and len(n.split()) >= 3:
                lineidx[n].add(wid)
        knownurls.update(s.get('url') for s in r.get('sources', []))
    groups = defaultdict(list)
    exclusions = []
    for r in inventory:
        if not r.get('genre_flags', {}).get('poetry'):
            continue
        year, death = r.get('source_year'), r.get('author_death_year')
        if r['url'] in knownurls:
            exclusions.append(dict(url=r['url'], reason='known_poetry_v6_source_url')); continue
        if (year is not None and not 1700 <= year <= 1900) or death is None or death > 1918:
            continue
        if re.search(r'эпиграммы на|памяти|биограф|переписк|рапорт|прозе|senilia|критик|памятник|перевод|из оперы|старообряд', r['title'], re.I):
            continue
        if r.get('genre_flags', {}).get('translation') or r.get('genre_flags', {}).get('reference'):
            continue
        groups[r['author_path']].append(r)
    # Diverse authors, prefer single titles before full collected editions.
    for group in groups.values():
        group.sort(key=lambda r: (bool(re.search('собрание|стихотворения|избранные', r['title'], re.I)), r['url']))
    selected = []
    for i in range(max(map(len, groups.values()), default=0)):
        for author in sorted(groups):
            if i < len(groups[author]) and len(selected) < args.limit:
                selected.append(groups[author][i])
    if args.existing_only or args.top_up_danilevsky or args.extra_author:
        downloaded_urls = {x['url'] for x in rows(AUDIT / 'downloads.jsonl')}
        selected = [r for r in inventory if r['url'] in downloaded_urls]
        if args.top_up_danilevsky:
            selected += [r for r in groups.get('d/danilewskij_g_p', []) if r['url'] not in downloaded_urls]
        for author_path in args.extra_author:
            selected += [r for r in groups.get(author_path, []) if r['url'] not in {x['url'] for x in selected}]
        selected = selected[:args.limit]
    results, candidates = [], []
    previous = {r['url']: r for r in rows(AUDIT / 'downloads.jsonl')}
    downloads = list(previous.values())
    for entry in selected:
        path = AUDIT / 'raw' / entry['filename']
        if not path.exists() and args.fetch:
            time.sleep(2.1)
            result = subprocess.run(['curl.exe', '--fail', '--location', '--silent', '--show-error', '--max-time', '50', '--max-filesize', '10000000', '--output', str(path), entry['url']], capture_output=True)
            if result.returncode:
                results.append(dict(url=entry['url'], status='fetch_failed', detail=result.stderr.decode(errors='replace')[:500])); continue
        if not path.exists():
            continue
        raw = path.read_bytes()
        if entry['url'] not in previous:
            downloads.append(dict(url=entry['url'], snapshot=str(path.relative_to(ROOT)), sha256=sha(raw), bytes=len(raw)))
            write_rows(AUDIT / 'downloads.jsonl', downloads)
        text, decisions = extract(raw)
        lines = [x for x in text.splitlines() if x.strip()]
        lengths = [len(x.split()) for x in lines]
        overlap = Counter(wid for line in lines for wid in lineidx.get(norm(line), ()))
        related = set(w for w, count in overlap.items() if count >= 2 or count / max(1, len(lines)) > .2)
        related.update(titleidx.get((norm(entry['author']).split(' ')[0], norm(entry['title'])), ()))
        if norm(text) in exact:
            related.add(exact[norm(text)])
        reasons = []
        if related: reasons.append('known_work_family_or_shared_lines_block_assignment')
        if len(lines) < 4 or statistics.median(lengths or [999]) > 11 or max(lengths or [999]) > 35:
            reasons.append('verse_form_or_boundary_review')
        if re.search(r'собрание|стихотворени[яй]|избранные|^поэзия$|надписи и послания', entry['title'], re.I) or any('multiple_headings' in x for x in decisions):
            reasons.append('collection_requires_individual_work_boundaries')
        if re.search(r'редакци|вариант|отрыв|фрагмент|из поэмы|из драматич', entry['title'], re.I):
            reasons.append('variant_or_fragment_excluded_from_ready_candidates')
        if entry.get('genre_flags', {}).get('essays_criticism') or entry.get('genre_flags', {}).get('drama'):
            reasons.append('mixed_genre_catalogue_flag_requires_review')
        if re.search(r'сатира на|[А-Я]\.\s*[А-Я]\.\s*\S+ или', entry['title']):
            reasons.append('catalogue_directory_not_reliable_work_authorship')
        if re.search(r'OCR|источник|издатель|типограф|примечан|комментари|подготов|©|copyright|https?://|[\[\]{}_^]|[ѣіѳѵ]', text, re.I):
            reasons.append('editorial_markup_or_historical_orthography_review')
        if not entry.get('source_year'):
            reasons.append('work_date_not_established_by_catalog')
        if not text:
            reasons.append('empty_extraction')
        row = dict(candidate_id='poetry_v7_' + sha(entry['url'])[:16], author=entry['author'], title=entry['title'], text=text,
                   source_url=entry['url'], snapshot=str(path.relative_to(ROOT)), snapshot_sha256=sha(raw), text_sha256=sha(text),
                   era_evidence={k: entry.get(k) for k in ['source_year', 'author_birth_year', 'author_death_year', 'period_labels', 'date_basis', 'catalog_url', 'catalog_sha256', 'catalog_entry']},
                   quality_status='candidate_only_not_accepted', split_status='unassigned_pending_work_family_and_cross_stage_review',
                   preliminary_status='blocked_known_family' if related else ('review_required' if reasons else 'automated_verse_candidate_requires_boundary_review'),
                   review_reasons=reasons, extraction_decisions=decisions, nonempty_lines=len(lines), words=len(text.split()),
                   related_known_works=[dict(work_id=w, author=knownmap[w]['author'], title=knownmap[w]['title'], split=assignments.get(w), shared_lines=overlap[w]) for w in sorted(related)])
        candidates.append(row)
        results.append({k: v for k, v in row.items() if k != 'text'})
        write_rows(OUT / 'candidates.jsonl', candidates)
        write_rows(AUDIT / 'review.jsonl', results)
    write_rows(AUDIT / 'excluded_known_sources.jsonl', exclusions)
    bounded = [dict(r, preliminary_status='manually_bounded_verse_candidate', extraction_decisions=r['extraction_decisions'] + ['full_text_first_last_and_metadata_inspected_2026_10_31']) for r in candidates if r['title'] == 'Брат и сестра' and r['author'].startswith('Данилевский') and not r['review_reasons']]
    # Individually inspected 2026-10-31: two complete poems before the explicitly
    # labelled Brodyaga excerpt. Keep excerpt, compiler biography, and other poems
    # with visible source defects in the parent review queue.
    aksakov = next((r for r in candidates if r['source_url'].endswith('/a/aksakow_i_s/text_0030.shtml')), None)
    if aksakov:
        for title, opening, closing, year in [
            ('Смотри! толпа людей нахмурившись стоит', 'Смотри! толпа людей нахмурившись стоит:', 'Работников из них составить полк лихой.', 1846),
            ('Зачем душа твоя смирна?', 'Зачем душа твоя смирна?\nЧем в этом мире ты утешен?', 'Дано лучей ее добыть!..', None),
        ]:
            start = aksakov['text'].index(opening)
            stop = aksakov['text'].index(closing, start) + len(closing)
            text = aksakov['text'][start:stop]
            hits = Counter(w for line in text.splitlines() for w in lineidx.get(norm(line), ()))
            if any(n >= 2 for n in hits.values()) or norm(text) in exact:
                continue
            row = dict(aksakov, candidate_id='poetry_v7_' + sha(aksakov['source_url'] + title)[:16], title=title,
                       text=text, text_sha256=sha(text), words=len(text.split()), nonempty_lines=sum(bool(x.strip()) for x in text.splitlines()),
                       preliminary_status='manually_bounded_verse_candidate', review_reasons=[], related_known_works=[],
                       extraction_decisions=aksakov['extraction_decisions'] + ['manually_inspected_complete_poem_literal_boundaries; heading/date and compiler footer excluded'],
                       individual_boundary_evidence={'parent_candidate_id': aksakov['candidate_id'], 'first_text': opening, 'last_text': closing, 'page_poem_year': year, 'author_lifespan': [1823, 1886]})
            bounded.append(row)
    segmented_review = []
    candidate_exact = {norm(r['text']) for r in bounded}
    for parent, title, text, year, evidence in individual_poems(candidates):
        nonempty = [x for x in text.splitlines() if x.strip()]
        lengths = [len(x.split()) for x in nonempty]
        hits = Counter(w for line in nonempty for w in lineidx.get(norm(line), ()))
        reasons = []
        if norm(text) in exact or any(n >= 2 for n in hits.values()):
            reasons.append('known_poetry_main_reserve_exact_or_line_family')
        if norm(text) in candidate_exact:
            reasons.append('duplicate_within_new_bounded_candidates')
        if not 4 <= len(nonempty) <= 1000 or statistics.median(lengths or [999]) > 10 or max(lengths or [999]) > 20:
            reasons.append('verse_form_boundary_review')
        if re.search(r'[_^{}\[\]ѣіѳѵ]|https?://|примечани|комментари|См\.:|\d{3,}', text, re.I):
            reasons.append('unresolved_editorial_markup_or_date_in_verse')
        if re.search(r'отрыв|фрагмент|перевод|из [А-Я]', title):
            reasons.append('fragment_translation_or_attribution_review')
        row = dict(parent, candidate_id='poetry_v7_' + sha(parent['source_url'] + title + str(year) + text)[:16], title=title,
                   text=text, text_sha256=sha(text), words=len(text.split()), nonempty_lines=len(nonempty),
                   preliminary_status='individually_bounded_verse_candidate' if not reasons else 'segmented_review_required',
                   review_reasons=reasons, related_known_works=[dict(work_id=w, title=knownmap[w]['title'], split=assignments.get(w), shared_lines=n) for w, n in hits.items() if n >= 2],
                   individual_boundary_evidence=evidence, split_guard_group='source_collection_' + sha(parent['source_url'])[:16],
                   era_evidence=dict(parent['era_evidence'], individual_work_year=year, individual_date_basis=evidence['kind']),
                   extraction_decisions=parent['extraction_decisions'] + ['individual_work_boundaries; own source date <=1900; source collection kept as indivisible split guard group'])
        segmented_review.append({k: v for k, v in row.items() if k != 'text'})
        if not reasons:
            bounded.append(row); candidate_exact.add(norm(text))
    # Invert only the small candidate shingle set, then scan every frozen main
    # and reserve work. This guards partial and punctuation-edited copies too.
    candidate_grams = defaultdict(set)
    gram_counts = {}
    for r in bounded:
        w = norm(r['text']).split(); grams = {' '.join(w[i:i+5]) for i in range(len(w)-4)}
        gram_counts[r['candidate_id']] = len(grams)
        for gram in grams: candidate_grams[gram].add(r['candidate_id'])
    shingle_matches = defaultdict(list)
    for old in known:
        w = norm(old['text']).split()
        hits = Counter(cid for gram in {' '.join(w[i:i+5]) for i in range(len(w)-4)} for cid in candidate_grams.get(gram, ()))
        for cid, count in hits.items():
            coverage = count / max(1, gram_counts[cid])
            if (count >= 3 and coverage >= .3) or (count >= 8 and coverage >= .1):
                shingle_matches[cid].append(dict(work_id=old['work_id'], title=old['title'], split=assignments.get(old['work_id']), shared_5grams=count, candidate_coverage=coverage))
    retained = []
    for r in bounded:
        if r['candidate_id'] in shingle_matches:
            r.update(preliminary_status='segmented_review_required', review_reasons=['known_main_reserve_5gram_family_overlap'], related_known_works=shingle_matches[r['candidate_id']])
            segmented_review.append({k: v for k, v in r.items() if k != 'text'})
        else:
            r['dedup_evidence'] = 'exact normalized text, >=2 shared verse lines, and normalized 5word overlap scanned against all poetry_v6 main+reserve works in every split'
            retained.append(r)
    bounded = retained
    write_rows(AUDIT / 'individual_work_review.jsonl', segmented_review)
    qc_path = AUDIT / 'full_text_qc.jsonl'
    if args.initialize_qc_after_full_reading:
        qc_rows = []
        for r in bounded:
            reasons = [reason for fragment, reason in QC_ISSUES.items() if fragment in r['text']]
            qc_rows.append(dict(candidate_id=r['candidate_id'], text_sha256=r['text_sha256'], author=r['author'], title=r['title'], source_url=r['source_url'],
                                status='needs_review' if reasons else 'recommended_usable',
                                reason='; '.join(reasons) if reasons else 'Full text read: coherent classical verse; no remaining prose/editorial intrusion or obvious source defect found.',
                                review_scope='All 114 bounded candidate texts read in full, including long merged Sevastyan; editorial inspection, not facsimile collation or measured training quality.',
                                admission_status='not_accepted_cross_stage_guard_pending'))
        write_rows(qc_path, qc_rows)
    qc = {r['candidate_id']: r for r in rows(qc_path)}
    proposed_train = []
    for r in bounded:
        r['canonical_author'] = 'Романов Константин Константинович' if r['author'] == 'К. Р.' else r['author']
        r['work_family_id'] = 'poetry_family_' + sha(norm(r['canonical_author']) + '\0' + norm(r['title']))[:16]
        r.setdefault('split_guard_group', 'source_collection_' + sha(r['source_url'])[:16])
        decision = qc.get(r['candidate_id'])
        if decision and decision['text_sha256'] == r['text_sha256']:
            r['full_text_qc'] = decision
            if decision['status'] == 'recommended_usable':
                proposed_train.append(dict(r, proposed_usage='train_only_if_root_cross_stage_guard_passes', split_status='unassigned_pending_root_prose_v7_heldout_guard'))
        else:
            r['full_text_qc'] = {'status': 'needs_review', 'reason': 'No matching content-hash-bound editorial decision.'}
    write_rows(OUT / 'proposed_train_candidates.jsonl', proposed_train)
    write_rows(OUT / 'bounded_candidates.jsonl', bounded)
    summary = dict(inventory_rows=len(inventory), selected_pages=len(selected), downloaded_pages=len(downloads), extracted_pages=len(candidates), known_poetry_main_reserve_works=len(known), statuses=dict(Counter(r['preliminary_status'] for r in candidates)), total_candidate_words=sum(r['words'] for r in candidates), accepted_works=0, split_assignments_created=0, training_started=False, scope='Candidates only. No frozen corpora or model weights modified. Collections require per-work segmentation; all candidates require cross-stage checks before admission.')
    summary.update(bounded_candidate_works=len(bounded), bounded_candidate_words=sum(r['words'] for r in bounded))
    summary.update(full_text_qc_counts=dict(Counter(r['full_text_qc']['status'] for r in bounded)), proposed_train_candidate_works=len(proposed_train), proposed_train_candidate_words=sum(r['words'] for r in proposed_train))
    (AUDIT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
