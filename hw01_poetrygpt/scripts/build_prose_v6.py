"""Versioned, source-pinned repairs of prose v5; no training or broad recollection.

V5 admission decisions remain inherited. A reviewed recovery whitelist repairs
specific false exclusions; contextual OCR checks repair only nonprinting format
bytes and quarantine unresolved paragraphs. Every deletion creates a boundary.
"""
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer
from build_prose_v3 import FragmentIndex, read_lines, write_json, write_lines, words
from build_prose_v5 import clean as clean_v5, strip_html

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/prose_v6'
AUDIT = ROOT / 'audit/prose_v6'
OLD = ROOT / 'data/prose_v5'
RAW = ROOT / 'audit/prose_v5'
C0 = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd\u200b\u2060]')
SOFT = re.compile(r'(?<!\w)[Ьь][\u0400-\u052f]{2,}')
PUNCT = re.compile(r'[\u0400-\u052f][\^_|`~/\\][\u0400-\u052f]')
DIGIT = re.compile(r'[\u0400-\u052f]\d|\d[\u0400-\u052f]')
WORD = re.compile(r'[^\W\d_]+', re.UNICODE)
LATIN_TO_CYR = dict(zip('ABCEHKMOPTXYaceopxy', 'АВСЕНКМОРТХУасеорху'))
CYR_TO_LATIN = {v:k for k,v in LATIN_TO_CYR.items()}
VOWELS = set('аеёиоуыэюяАЕЁИОУЫЭЮЯѣѢ')
RECOVER = {
    'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0090.shtml': [3358, 3962],
    'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0020.shtml': [197],
    'http://az.lib.ru/t/turgenew_i_s/text_0260.shtml': [59],
    'http://az.lib.ru/p/pisemskij_a/text_0170.shtml': [547],
    'http://az.lib.ru/b/boborykin_p_d/text_1894_neizlechimye.shtml': [117],
    'http://az.lib.ru/l/leskow_n_s/text_1865_oboydennye.shtml': [3473],
    'http://az.lib.ru/a/aksakow_s_t/text_0004.shtml': [259],
    'http://az.lib.ru/a/aksakow_s_t/text_0010.shtml': [223],
    'http://az.lib.ru/k/krestowskij_w_w/text_0110.shtml': [3678],
    'http://az.lib.ru/k/karamzin_n_m/text_1838_detsky_karamzin_oldorfo.shtml': [509],
}


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode('utf8')).hexdigest()


def normalize_source_paragraph(p):
    p = unicodedata.normalize('NFC', p).translate(str.maketrans('ѢѣІіѲѳѴѵ', 'ЕеИиФфИи'))
    p = re.sub(r'(?<=[А-Яа-яЁё])[ъЪ]\b', '', p)
    p = re.sub(r'\{[^{}]*\}|\[\d+\]', '', p)
    p = re.sub(r'(?<!-)--(?!-)', '—', p)
    return re.sub(r'[ \t\u00a0]+', ' ', p).strip()


def source_view(entry):
    data = (RAW/'raw'/entry['filename']).read_bytes()
    assert sha(data) == entry['sha256']
    html = data.decode('cp1251').replace('\r', '')
    start = re.search(r'<!--[^>]*Собственно произведение[^>]*-->', html).end()
    stop = html.find('</noindex>', start)
    raw = [re.sub(r'[ \t\u00a0]+', ' ', p).strip() for p in strip_html(html[start:stop]).splitlines()]
    raw = [p for p in raw if p]
    accepted, history = clean_v5(entry)
    first = history[0]['paragraphs']
    last = next((op['paragraph_from'] for op in history if op['reason'] == 'editorial_tail_excluded'), len(raw))
    positions, normalized = [], []
    for i in range(first, last):
        p = normalize_source_paragraph(raw[i])
        if not re.search('[А-Яа-яЁё]', p):
            continue
        if re.fullmatch(r'(?:1[789]\d\d)(?:\s*[—-]\s*1[789]\d\d)?\s*(?:г\.?|года)?\.?', p):
            continue
        positions.append(i); normalized.append(p)
    assert len(accepted) == len(normalized), entry['url']
    assert all(p is None or p == normalized[i] for i, p in enumerate(accepted)), entry['url']
    return raw, accepted, normalized, positions


def new_row(parent, text, parents, reason, tokenizer, extra=None):
    row = dict(parent)
    row['text'] = text; row['text_sha256'] = sha(text)
    row['work_id'] = 'prose_v6_' + sha(parent['family_id'] + parent['sources'][0]['url'] + text)[:24]
    row['token_count'] = len(tokenizer.encode(text, add_special_tokens=False).ids)
    row['quality'] = {**parent.get('quality', {}), 'v6_parent_work_ids': sorted(set(parents)),
                      'v6_revision': reason, 'is_excerpt': True, **(extra or {})}
    return row


def recover_sources(old, entries, tokenizer, ledger, packet):
    groups = defaultdict(list)
    for row in old:
        groups[row['sources'][0]['url']].append(row)
    replacements = {}
    for url, raw_ids in RECOVER.items():
        parents = groups[url]; entry = entries[url]
        raw, accepted, normalized, raw_positions = source_view(entry)
        owners, retained = {}, [None] * len(accepted)
        for row in parents:
            parts = row['text'].split('\n\n')
            lo = row['quality']['paragraph_from']; hi = row['quality']['paragraph_to'] + 1
            matches = [i for i in range(lo, hi) if accepted[i:i+len(parts)] == parts]
            assert len(matches) == 1, (url, row['work_id'], matches)
            for i, p in enumerate(parts, matches[0]):
                assert retained[i] is None, (url, i)
                retained[i] = p; owners[i] = row['work_id']
        original_count = Counter(p for r in parents for p in r['text'].split('\n\n'))
        restored = []
        for raw_i in raw_ids:
            i = raw_positions.index(raw_i)
            assert accepted[i] is None and retained[i] is None, (url, raw_i)
            retained[i] = normalized[i]; restored.append(normalized[i])
            ledger.append({'action': 'restore_reviewed_authorial_paragraph', 'url': url,
                           'raw_sha256': entry['sha256'], 'raw_paragraph': raw_i,
                           'source_text': raw[raw_i], 'before_sha256': None,
                           'after_sha256': sha(normalized[i]), 'after': normalized[i],
                           'reason': 'Authorial narrative/dialogue/numbered heading, not source metadata. Reviewed narrow whitelist.'})
            packet.append({'kind': 'recovery', 'url': url, 'raw_sha256': entry['sha256'],
                           'raw_paragraph': raw_i, 'raw_neighbors': raw[max(0,raw_i-1):raw_i+2],
                           'after': normalized[i], 'split': parents[0]['split']})
        assert Counter(p for p in retained if p is not None) == original_count + Counter(restored)
        chunks, current, indices = [], [], []
        for i, p in enumerate(retained + [None]):
            if p is not None:
                current.append(p); indices.append(i)
            elif current:
                text = '\n\n'.join(current)
                previous = [r for r in parents if r['text'] == text]
                if previous:
                    assert len(previous) == 1
                    chunks.append(previous[0])
                elif len(words(text)) >= 20:
                    ids = [owners[j] for j in indices if j in owners]
                    chunks.append(new_row(parents[0], text, ids, 'source_verified_recovery', tokenizer,
                        {'v6_source_raw_paragraph_range': [raw_positions[indices[0]], raw_positions[indices[-1]]],
                         'v6_raw_sha256': entry['sha256']}))
                else:
                    ledger.append({'action': 'defer_isolated_short_recovery', 'url': url,
                                   'text': text, 'text_sha256': sha(text)})
                current, indices = [], []
        replacements[url] = chunks
        print('Recovered source', url, 'paragraphs', len(raw_ids), flush=True)
    emitted, result = set(), []
    for r in old:
        url = r['sources'][0]['url']
        if url in replacements:
            if url not in emitted:
                result.extend(replacements[url]); emitted.add(url)
        else:
            result.append(r)
    return result


def repair_format_controls(p):
    """Delete only source-present nonprinting bytes in constrained contexts.

    No printable character is inserted/replaced. U+0016 (often a lost numero)
    and intraword corruption remain unresolved and are quarantined.
    """
    removed = []
    for m in C0.finditer(p):
        i = m.start(); ch = m.group(); before = p[i-1] if i else ''; after = p[i+1:i+2]
        safe = ((ch == '\x01' and before in VOWELS and not after.isdigit())
                or (ch == '\x07' and (not before or not before.isalnum()) and after != '1')
                or (ch == '\x0e' and i == 0)
                or (ch == '\x00' and before.isalpha() and after.isspace())
                or (ch == '\x1a' and after in ('—', '–'))
                or (ch in ('\u200b', '\u2060') and (not before or before.isspace() or not after or after.isspace())))
        if safe:
            removed.append(i)
    return ''.join(ch for i, ch in enumerate(p) if i not in set(removed)), removed


def issues(p):
    result = {name: [m.group() for m in pattern.finditer(p)] for name, pattern in
            [('initial_soft_sign', SOFT), ('intraword_punctuation', PUNCT),
             ('unresolved_control_or_replacement', C0), ('digit_adjacent_letter', DIGIT)] if pattern.search(p)}
    mixed = ([m.group() for m in WORD.finditer(p) if re.search('[A-Za-z]',m.group()) and re.search('[\u0400-\u052f]',m.group())]
             if re.search('[A-Za-z]', p) else [])
    if mixed:
        result['unresolved_mixed_script_word'] = mixed
    if re.search(r'(?<!\w)ІІьер\w*', p):
        result['reviewed_ambiguous_name_ocr'] = ['ІІьер']
    return result


def repair_homoglyphs(p):
    if not re.search('[A-Za-z]', p):
        return p, []
    replacements = []
    def replace(m):
        token=m.group()
        if not re.search('[A-Za-z]', token) or not re.search('[\u0400-\u052f]', token):
            return token
        latin=[c for c in token if re.fullmatch('[A-Za-z]',c)]
        cyr=[c for c in token if '\u0400' <= c <= '\u052f']
        if not latin or not cyr:
            return token
        if re.fullmatch('[IVXLCDMІХСМ]+', token):
            new=token.translate(str.maketrans({'І':'I','Х':'X','С':'C','М':'M'}))
            reason='Roman numeral visual homoglyphs'
        elif all(c in LATIN_TO_CYR for c in latin) and any(c not in CYR_TO_LATIN for c in cyr):
            new=''.join(LATIN_TO_CYR.get(c,c) for c in token)
            reason='Latin visual homoglyphs inside Cyrillic word'
        else:
            return token
        replacements.append({'offset':m.start(),'before':token,'after':new,'reason':reason})
        return new
    return WORD.sub(replace,p),replacements


def editorial_boundary_flag(p, i, n):
    # Context and multiple bibliographic cues, never a blacklist of "printing".
    opening = i < 8 and re.search(r'роман русского писателя|(?:в книгу|в сборник|в настоящее издание) вошли', p, re.I)
    tail = i >= n-5 and len(p) < 650 and re.search(r'Литературная [Гг]азета|Источник текста|Издательство|Печатается по', p)
    bibliographic = re.search(r'\b(?:18|19|20)\d\d\b.*(?:№|С\.|\bкн\.|Изд\.)', p)
    return bool(opening or (tail and bibliographic))


def clean_rows(rows, tokenizer, ledger, dispositions):
    result = []
    for row in rows:
        # RSD legacy excerpts preserve source paragraphs as single LF; Lib.ru
        # excerpts use double LF. Keep this original separator on every rewrite.
        separator = '\n' if row['sources'][0].get('dataset') == 'RSD' else '\n\n'
        parts = row['text'].split(separator)
        output, changed = [], False
        for i, p in enumerate(parts):
            repaired, removed = repair_format_controls(p)
            repaired, homoglyphs = repair_homoglyphs(repaired)
            flags = issues(repaired)
            if editorial_boundary_flag(repaired, i, len(parts)):
                flags['contextual_bibliographic_block'] = [repaired[:180]]
            if flags or removed or homoglyphs:
                action = ('quarantine_paragraph' if flags else 'repair_visual_homoglyphs' if homoglyphs else 'repair_nonprinting_format_bytes')
                entry = {'action': action, 'work_id': row['work_id'], 'split': row['split'],
                         'url': row['sources'][0]['url'], 'source_raw_sha256': row['sources'][0].get('raw_sha256'),
                         'paragraph': i, 'before': p, 'before_sha256': sha(p),
                         'after': None if flags else repaired, 'after_sha256': None if flags else sha(repaired),
                         'flags': flags, 'removed_control_offsets': removed, 'homoglyph_replacements':homoglyphs,
                         'reason': 'Unresolved source defect; no guessed letter replacement.' if flags else
                                   'Source-pinned constrained formatting or visual homoglyph repair; no lexical reconstruction or foreign transliteration.'}
                ledger.append(entry); dispositions.append(entry)
                changed = True
            output.append(None if flags else repaired)
        if not changed:
            result.append(row); continue
        chunks, current = [], []
        for p in output + [None]:
            if p is not None:
                current.append(p)
            elif current:
                chunks.append(separator.join(current)); current = []
        for text in chunks:
            if len(words(text)) < 40:
                ledger.append({'action': 'quarantine_short_remainder_after_gap', 'work_id': row['work_id'],
                               'text': text, 'text_sha256': sha(text), 'split': row['split']})
                continue
            result.append(new_row(row, text, [row['work_id']], 'contextual_ocr_review', tokenizer,
                                  {'source_paragraph_range_is_parent_excerpt': True}))
    return result


def guard_changed_train(rows, old_ids, tokenizer, ledger):
    prose, poetry = FragmentIndex(20), FragmentIndex(12)
    for row in rows:
        if row['split'] != 'train':
            prose.add('prose/'+row['work_id'], row['text'])
    for split in ('validation', 'test'):
        for row in read_lines(ROOT/f'data/poetry_v5/{split}.jsonl'):
            poetry.add('poetry/'+row['work_id'], row['text'])
    result = []
    for row in rows:
        if row['split'] != 'train' or row['work_id'] in old_ids:
            result.append(row); continue
        hits = list(prose.matches(row['text'])) + list(poetry.matches(row['text']))
        if hits:
            # New material only: conservative whole excerpt quarantine; no risk
            # of leaving adjacent unmatched verse lines. Root audits every row.
            ledger.append({'action': 'quarantine_changed_excerpt_heldout_overlap', 'work_id': row['work_id'],
                           'text_sha256': row['text_sha256'], 'matches': hits[:20]})
        else:
            result.append(row)
    return result


def main():
    OUT.mkdir(exist_ok=True); AUDIT.mkdir(exist_ok=True)
    manifest_path = RAW/'download_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    entries = {e['url']:e for e in manifest['files']}
    inputs = [OLD/'corpus.jsonl', OLD/'train.jsonl', OLD/'validation.jsonl', OLD/'test.jsonl',
              manifest_path, RAW/'transformations.jsonl', ROOT/'artifacts/tokenizers_v3/selected/tokenizer.json']
    inputs += [ROOT/f'data/poetry_v5/{s}.jsonl' for s in ('validation','test')]
    hashes = {p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in inputs}
    tokenizer = Tokenizer.from_file(str(ROOT/'artifacts/tokenizers_v3/selected/tokenizer.json'))
    old = read_lines(OLD/'corpus.jsonl'); by_id = {r['work_id']:r for r in old}
    ledger, packet, dispositions = [], [], []
    assert repair_homoglyphs('Mонах Неrr Рluck ChХre cа')[0] == 'Монах Неrr Рluck ChХre cа'
    assert issues(repair_homoglyphs('Неrr Рluck ChХre cа')[0])
    assert issues(repair_homoglyphs('chиre')[0])
    assert not issues('ѣсть, очи, сей; ma chère; типография находилась в деревне')
    assert not editorial_boundary_flag('1. В то время ученики приступили к Иисусу — читал он.', 10, 20)
    rows = recover_sources(old, entries, tokenizer, ledger, packet)
    rows = clean_rows(rows, tokenizer, ledger, dispositions)
    # Source contains the same prose twice; only the next chapter headings differ.
    # Keep the first occurrence, never silently modify the frozen parent heldout.
    duplicate_id = 'prose_dostoevsky_besy_s0062'
    retained_id = 'prose_dostoevsky_besy_s0055'
    duplicate = next(r for r in rows if r['work_id'] == duplicate_id)
    retained = next(r for r in rows if r['work_id'] == retained_id)
    assert duplicate['split'] == retained['split'] == 'validation'
    assert duplicate['family_id'] == retained['family_id']
    assert duplicate['text'].splitlines()[:-2] == retained['text'].splitlines()[:-2]
    assert duplicate['text_sha256'] == '46c6942096835f675943a540f60b0886271bd266bf052b91f50a35e42ebcd70d'
    ledger.append({'action':'remove_reviewed_near_duplicate', 'work_id':duplicate_id,
                   'retained_work_id':retained_id, 'split':'validation',
                   'before_sha256':duplicate['text_sha256'], 'after_sha256':None,
                   'url':duplicate['sources'][0]['url'], 'source_lines':[2180,2268],
                   'reason':'Identical full prose; only two trailing next-chapter headings differ.',
                   'evidence':'audit/prose_v6/reviewed_duplicate_evidence.json'})
    rows = [r for r in rows if r['work_id'] != duplicate_id]
    rows = guard_changed_train(rows, set(by_id), tokenizer, ledger)
    families = {}; exact = {}
    for row in rows:
        assert families.setdefault(row['family_id'], row['split']) == row['split']
        assert row['split'] == next(r['split'] for r in old if r['family_id'] == row['family_id'])
        assert sha(row['text']) == row['text_sha256']
        digest = sha(' '.join(words(row['text'])))
        assert digest not in exact, ('normalized_duplicate', exact.get(digest), row['work_id'])
        exact[digest] = row['work_id']
        assert not issues(row['text']), row['work_id']
    assert len({r['work_id'] for r in rows}) == len(rows)
    changed_heldout = [r for r in rows if r['split'] != 'train' and r['work_id'] not in by_id]
    write_lines(OUT/'corpus.jsonl', rows)
    write_lines(OUT/'metadata.jsonl', [{k:v for k,v in r.items() if k!='text'} for r in rows])
    for split in ('train','validation','test'):
        write_lines(OUT/f'{split}.jsonl', [{'work_id':r['work_id'],'text':r['text']} for r in rows if r['split']==split])
    write_json(OUT/'split_assignments.json', {r['work_id']:{'split':r['split'],'family_id':r['family_id']} for r in rows})
    write_lines(AUDIT/'changes.jsonl', ledger)
    write_json(AUDIT/'flag_dispositions.json', dispositions)
    write_json(AUDIT/'heldout_changes.json', {'version':'v6','model_outputs_used':False,
        'new_or_changed':[{'work_id':r['work_id'],'split':r['split'],'family_id':r['family_id'],
                           'quality':r['quality'],'text_sha256':r['text_sha256']} for r in changed_heldout],
        'removed_v5_ids':[r['work_id'] for r in old if r['split']!='train' and r['work_id'] not in {x['work_id'] for x in rows}]})
    # Fixed hash sample of unflagged retained records: negative controls, not
    # evidence that every unflagged paragraph is correct. Distinct families.
    flagged_ids = {d['work_id'] for d in dispositions}; seen = set()
    for row in sorted(old, key=lambda r:sha('prose_v6_negative_seed:'+r['work_id'])):
        if row['work_id'] in flagged_ids or row['family_id'] in seen or issues(row['text']):
            continue
        parts=row['text'].split('\n\n'); i=int(sha(row['work_id'])[:8],16)%len(parts)
        packet.append({'kind':'unflagged_negative_control','work_id':row['work_id'],'family_id':row['family_id'],
                       'split':row['split'],'url':row['sources'][0]['url'],'paragraph':i,'text':parts[i],
                       'context':parts[max(0,i-1):i+2]})
        seen.add(row['family_id'])
        if len(seen)==30:break
    for d in sorted(dispositions, key=lambda d:sha(d['work_id']+str(d['paragraph'])))[:30]:
        packet.append({'kind':'flag_disposition_sample',**d})
    write_json(AUDIT/'independent_review_packet.json', packet)
    stats={}
    for split in ('train','validation','test'):
        subset=[r for r in rows if r['split']==split]
        tokens=sum(r['token_count'] if 'token_count' in r else len(tokenizer.encode(r['text'],add_special_tokens=False).ids) for r in subset)
        stats[split]={'records':len(subset),'families':len({r['family_id'] for r in subset}),
                      'author_labels':len({r['author'] for r in subset}),'content_tokens':tokens,
                      'authors_characters':dict(Counter({a:sum(len(r['text']) for r in subset if r['author']==a) for a in {r['author'] for r in subset}})),
                      'source_characters':dict(Counter({s:sum(len(r['text']) for r in subset if r['sources'][0]['dataset']==s) for s in {r['sources'][0]['dataset'] for r in subset}}))}
        print('Counted',split,tokens,flush=True)
    assert all(sha((ROOT/p).read_bytes())==h for p,h in hashes.items()), 'Frozen input changed'
    report={'version':'prose_v6','passed':True,'training_started':False,'input_sha256':hashes,
            'output_sha256':{p.name:sha(p.read_bytes()) for p in OUT.glob('*.json*')},
            'script_sha256':sha(Path(__file__).read_bytes()),'stats':stats,
            'actions':dict(Counter(x['action'] for x in ledger)),
            'unchanged_v5_records':sum(r['work_id'] in by_id and r==by_id[r['work_id']] for r in rows),
            'changed_heldout_records':len(changed_heldout),
            'recovery_candidates':sum(len(v) for v in RECOVER.values()),
            'isolated_recovery_candidates_deferred':sum(x['action']=='defer_isolated_short_recovery' for x in ledger),
            'regression_probes_passed':5,
            'limits':['V5 admission and most rejected paragraphs are inherited, not exhaustively reclassified.',
                      'C0 repair preserves printable characters; original accent glyphs are not reconstructed.',
                      'Unresolved flags quarantine paragraphs and short remainders, not guessed characters.',
                      'Root must run complete cross-stage v6 heldout audit before packing.',
                      'Fixed-hash samples are audit aids, not a corpus-wide error-rate estimate.',
                      'No generic Cyrillic-to-Latin repair: ambiguous mixed foreign text is quarantined.',
                      'Latin-to-Cyrillic repair requires a Cyrillic non-homoglyph anchor; Roman numerals handled separately.']}
    retention = {}
    for key in ('author','genre'):
        before = Counter(); after = Counter()
        for r in old:before[r.get(key,'unknown')] += len(r['text'])
        for r in rows:after[r.get(key,'unknown')] += len(r['text'])
        retention[key] = {a:{'v5_characters':before[a],'v6_characters':after[a],
                             'delta_characters':after[a]-before[a],
                             'retention_ratio':after[a]/before[a]} for a in before}
    before = Counter(); after = Counter()
    for r in old:before[r['sources'][0]['dataset']] += len(r['text'])
    for r in rows:after[r['sources'][0]['dataset']] += len(r['text'])
    retention['source'] = {a:{'v5_characters':before[a],'v6_characters':after[a],
                              'delta_characters':after[a]-before[a],
                              'retention_ratio':after[a]/before[a]} for a in before}
    write_json(AUDIT/'retention.json', retention)
    write_json(AUDIT/'report.json', report)
    print(json.dumps({'passed':True,'stats':{s:{k:v for k,v in x.items() if not isinstance(v,dict)} for s,x in stats.items()},'actions':report['actions']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
