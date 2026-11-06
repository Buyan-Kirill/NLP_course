"""Defensive Lib.ru extraction; no corpus writes or split assignment.

extract(entry) returns (paragraphs, ledger, source_flags). Each retained paragraph
and None boundary has its original parsed block_index in the ledger. Paths are
project-relative. Reviewed boundaries are accepted only for their pinned bytes.
"""
import hashlib
import html
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

from build_prose_v6 import issues, repair_format_controls, repair_homoglyphs

ROOT = Path(__file__).resolve().parents[1]
CYR = re.compile(r'[А-Яа-яЁё]')
META = re.compile(r'https?://|www\.|\bOCR\b|ISBN|©|электронная (?:версия|библиотека)|'
                  r'подготовка текста|сканирован|оцифров|^Источник\s*:|^Скан\s*:|'
                  r'^Форматирование|^Правка\s*:|^Сверка|^Тираж\s+\d|^Подписано к печати|'
                  r'^Сдано в набор|Собр\.\s*соч\.|собрание сочинений|'
                  r'^Издание (?:второе|третье|подготов)|^Серия [«"]|'
                  r'Гослитиздат|Государственное издательство|изд-во|'
                  r'^Повести и рассказы .*Произведения разных годов', re.I)
TAIL = re.compile(r'^(?:КОММЕНТАРИИ|ПРИМЕЧАНИЯ(?: И КОММЕНТАРИИ)?|СНОСКИ|'
                  r'УСЛОВНЫЕ (?:ОБОЗНАЧЕНИЯ|СОКРАЩЕНИЯ)|УКАЗАТЕЛЬ (?:ИМЕН|ИМЁН)|'
                  r'ИМЕННОЙ УКАЗАТЕЛЬ|СПИСОК ИЛЛЮСТРАЦИЙ|ОТ РЕДАКЦИИ|ПОСЛЕСЛОВИЕ)[\s.:*]*$', re.I)
INTRO = re.compile(r'вступительн\w*\s+(?:стать|очерк)|предисловие\s+(?:редак|издат)|'
                   r'подготовка текста и примечания|в книгу вошли|в сборник вошли|'
                   r'в настоящее издание вошли', re.I)
BIB = re.compile(r'^(?:М[.,:]|Л[.,:]|Москва|Санкт-Петербург|СПб[.,:]).*\b(?:1[89]\d\d|20\d\d)\b|'
                 r'^Том\s+(?:перв|втор|трет|четв|пят|шест|седьм|восьм|девят|десят|одиннадцат|двенадцат)\w*[.:]|'
                 r'^(?:Том|Т\.)\s*\d.*(?:стр\.|с\.)|^\d\s+\d\s+\d\s+\d$|'
                 r'^(?:Художественный |Технический )?(?:редактор|корректор)\s+[А-Я]\.', re.I)
CREDIT = re.compile(r'^(?:под редакцие[юйю]|Составление и подготовка текста|Подготовка текста|'
                    r'Сост\.,\s*подгот\.|Художник\s+[А-Я]\.|Иллюстрации и оформление)', re.I)
CONTENTS = re.compile(r'^(?:СОДЕРЖАНИЕ|ОГЛАВЛЕНИЕ|УКАЗАТЕЛЬ (?:ИМЕН|ИМЁН|СОБСТВЕННЫХ ИМЕН)|'
                      r'ИМЕННОЙ УКАЗАТЕЛЬ|БИОГРАФИЧЕСКИЙ СЛОВАРЬ ЛИЧНЫХ ИМЕН)[\s.:]*$', re.I)


def journal_footer(p, i, total):
    """Quoted publication title + issue and year, in the final source blocks."""
    return bool(i >= max(0, total - 8) and len(p) < 220
                and re.search(r'\b(?:17|18|19|20)\d\d\b', p)
                and re.fullmatch(r'["«][^"»]{2,100}["»][\s,.:]*(?:\d{4}[\s,.:]*)?'
                                 r'(?:(?:No){1,2}|№{1,2}|кн\.)\s*[IVX\d\s.,—–-]+' , p, re.I))


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode('utf8')).hexdigest()


class Blocks(HTMLParser):
    """Block boundaries from HTML; PRE indentation separates wrapped paragraphs."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.buffer, self.pre, self.skip = [], [], False, 0

    def flush(self):
        text = ''.join(self.buffer)
        self.buffer = []
        if self.pre:
            # A single physical line can be a soft wrap. New indented lines and
            # blank lines are source paragraph boundaries; never sentence-split.
            parts = re.split(r'\n[ \t]*\n+|\n(?=[ \t]{3,}\S)', text)
        else:
            parts = [text]
        for p in parts:
            p = re.sub(r'\s+', ' ', p).strip()
            if p:
                self.blocks.append(p)

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'sup'):
            self.skip += 1
            return
        if self.skip:
            return
        if tag == 'pre':
            self.flush(); self.pre = True
        elif tag in ('dd', 'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'hr'):
            self.flush()
        elif tag == 'br':
            if self.pre:
                self.buffer.append('\n')
            else:
                self.flush()
        elif tag not in ('a', 'b', 'i', 'u', 's', 'strong', 'em', 'font', 'span',
                         'center', 'dir', 'dl', 'dt', 'ul', 'ol', 'blockquote',
                         'small', 'big', 'sub', 'img', 'table', 'tbody', 'tr', 'td'):
            # Editorial expansions like Б<елозерская> are not HTML formatting.
            # Keep their visible markup so the annotation check quarantines them.
            if not re.fullmatch(r'xxx\d+', tag):
                self.buffer.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'sup') and self.skip:
            self.skip -= 1
            return
        if self.skip:
            return
        if tag == 'pre':
            self.flush(); self.pre = False
        elif tag in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'):
            self.flush()

    def handle_data(self, data):
        if not self.skip:
            self.buffer.append(data)


def normalize(p):
    p = unicodedata.normalize('NFC', p).translate(str.maketrans('ѢѣІіѲѳѴѵ', 'ЕеИиФфИи'))
    p = re.sub(r'(?<=[А-Яа-яЁё])[ъЪ]\b', '', p)
    p = re.sub(r'\[\d+\]|\{\d+\}', '', p)
    p = re.sub(r'(?<!-)--(?!-)', '—', p)
    return re.sub(r'\s+', ' ', p).strip()


def extract(entry):
    data = (ROOT / entry['path']).read_bytes()
    if sha(data) != entry['sha256']:
        raise ValueError('Raw source hash mismatch: ' + entry['path'])
    # A damaged byte excludes its paragraph, not the entire otherwise readable book.
    declared = re.search(br'charset\s*=\s*["\x27]?([A-Za-z0-9_-]+)', data[:5000], re.I)
    declared = declared[1].decode('ascii').lower() if declared else None
    aliases = {'windows-1251':'cp1251','cp1251':'cp1251','koi8-r':'koi8-r','utf-8':'utf-8-sig'}
    if declared in aliases:
        encoding = aliases[declared]
        source = data.decode(encoding, errors='replace')
    else:
        try:
            source = data.decode('utf-8-sig')
            encoding = 'utf-8'
        except UnicodeDecodeError:
            source = data.decode('cp1251', errors='replace')
            encoding = 'cp1251'
    source = source.replace('\r\r\n', '\n').replace('\r\n', '\n').replace('\r', '\n')
    ledger = [{'action': 'decode', 'encoding': encoding, 'raw_sha256': entry['sha256']}]
    flags = []
    opening = re.search(r'<!--[^>]*Собственно произведение[^>]*-->', source, re.I)
    if not opening:
        return [], ledger, ['missing_main_work_marker']
    end = source.find('</noindex>', opening.end())
    if end < 0:
        return [], ledger, ['missing_main_work_end']
    body = source[opening.end():end]
    # Only numeric anchors are known note markers. Preserve other links' text.
    body = re.sub(r'<a\b[^>]*href=["\']?#[^>]*>\s*(?:\d+|\[\d+\]|\*)\s*</a>', '', body, flags=re.I)
    parser = Blocks(); parser.feed(body); parser.flush()
    raw = parser.blocks
    exclusion_path = ROOT / 'audit/web_v7_quality/reviewed_paragraph_exclusions.json'
    exclusion_policies = (json.loads(exclusion_path.read_text(encoding='utf8'))
                          if exclusion_path.exists() else [])
    source_exclusions = [p for p in exclusion_policies if p['url'] == entry.get('url')]
    if source_exclusions and any(p['raw_sha256'] != entry['sha256'] for p in source_exclusions):
        return [], ledger, ['reviewed_paragraph_exclusion_source_mismatch']
    excluded_hashes = {h for policy in source_exclusions for h in policy['excluded_paragraph_sha256']}
    if excluded_hashes:
        available = {sha(normalize(p)) for p in raw}
        if not excluded_hashes <= available:
            return [], ledger, ['reviewed_paragraph_exclusion_anchor_mismatch']
        ledger.append({'action': 'source_pinned_paragraph_exclusion_policy',
                       'policy': str(exclusion_path.relative_to(ROOT)),
                       'normalized_paragraph_sha256': sorted(excluded_hashes)})
    letter_collection = bool(entry.get('genre_flags', {}).get('letters') and
                             re.search(r'переписка|письма|том\s+\d', entry.get('title',''), re.I))
    scholarly = re.compile(r'^(?:Печатается по|Датируется|Ответ на (?:два |три )?письм|'
                           r'Письмо .*не сохранилось|В автографе|Впервые (?:напечатано|опубликовано)|Редакционные варианты)', re.I)
    apparatus = [i for i,p in enumerate(raw) if scholarly.search(p)]
    start = next((i for i, p in enumerate(raw) if len(p) >= 80 and len(CYR.findall(p)) >= 45
                  and not META.search(p) and not BIB.search(normalize(p)) and not INTRO.search(p)
                  and not (len(p) < 650 and CREDIT.search(normalize(p)))), None)
    if start is None:
        return [], ledger, ['no_substantial_prose']
    reviewed_path = ROOT / 'audit/web_v7_quality/reviewed_boundaries.json'
    reviewed = next((p for p in json.loads(reviewed_path.read_text(encoding='utf8'))
                     if p['url'] == entry.get('url') and p['raw_sha256'] == entry['sha256']), None) if reviewed_path.exists() else None
    reviewed_end = len(raw)
    if reviewed:
        starts = [i for i,p in enumerate(raw) if normalize(p) == reviewed['start_anchor']]
        ends = [i for i,p in enumerate(raw) if normalize(p) == reviewed['end_anchor']] if reviewed.get('end_anchor') else [len(raw)]
        if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
            return [], ledger, ['reviewed_v7_boundary_anchor_mismatch']
        start, reviewed_end = starts[0], ends[0]
        ledger.append({'action':'source_pinned_v7_boundaries', 'block_from':start,
                       'block_to_exclusive':reviewed_end, 'policy':str(reviewed_path.relative_to(ROOT))})
    # Contents and indexes outside pinned authorial bounds cannot disqualify a
    # reviewed rescue. Unbounded letter editions require deliberate attribution
    # and boundaries before their table of contents can become training prose.
    if letter_collection:
        inspected = list(enumerate(raw[start:reviewed_end], start)) if reviewed else list(enumerate(raw))
        apparatus = [i for i, p in inspected if scholarly.search(normalize(p))]
        structures = [i for i, p in inspected if CONTENTS.fullmatch(normalize(p))]
        if len(apparatus) >= 3 or structures:
            ledger.append({'action': 'defer_unbounded_letter_edition',
                           'scholarly_blocks': apparatus[:100], 'structural_blocks': structures[:100]})
            return [], ledger, ['letter_edition_requires_reviewed_boundaries']
    # Reuse reviewed source policies by byte hash and a unique full narrative
    # anchor, independent of the older cleaner's paragraph numbering.
    policies = ROOT / 'audit/prose_v5/reviewed_source_prefixes.json'
    policy = next((p for p in json.loads(policies.read_text(encoding='utf8'))
                   if p['raw_sha256'] == entry['sha256']), None) if policies.exists() else None
    if reviewed:
        pass
    elif policy:
        anchor = normalize(policy['first_narrative_or_authorial_preface'])
        positions = [i for i, p in enumerate(raw) if normalize(p) == anchor]
        if len(positions) != 1:
            return [], ledger, ['reviewed_prefix_anchor_mismatch']
        start = positions[0]
        ledger.append({'action': 'pinned_reviewed_prefix', 'block_index': start,
                       'policy_path': str(policies.relative_to(ROOT)), 'anchor_sha256': sha(anchor)})
    else:
        # A modern scholarly introduction may read like clean historical prose.
        # Detect attribution in frontmatter and prose before admitting any body.
        head = '\n'.join(raw[:min(len(raw), start + 4)])
        surname = entry.get('author', '').split(' ')[0].strip('.,')
        critical_start = (len(surname) > 3 and surname.lower() in raw[start].lower()
                          and re.search(r'воспоминания|произведения|творчество|писател[ья]|литературн', raw[start], re.I)
                          and re.search(r'издавна пользуются|вошли|в настоящ|в предлагаем|в истории|прошлого века', raw[start], re.I))
        intro_positions = [i for i, p in enumerate(raw[:start]) if INTRO.search(p)]
        # Bibliographic credits can name an introduction absent from this page.
        # A subsequent chapter or authorial-preface heading bounds those credits.
        bounded_credits = bool(intro_positions) and any(
            re.fullmatch(r'(?:ГЛАВА|ЧАСТЬ|КНИГА)\s+\S+|ПРЕДИСЛОВИЕ(?: АВТОРА)?', p, re.I)
            for p in raw[intro_positions[-1] + 1:start])
        if (INTRO.search(head) and not bounded_credits) or critical_start:
            ledger.append({'action': 'defer_unbounded_modern_intro', 'block_index': start,
                           'opening': raw[start][:700]})
            return [], ledger, ['deferred_modern_intro']
        # Review queue only: these markers do not date the whole work. Inspect
        # the prospective prose opening, after bibliographic frontmatter; pinned
        # source bounds have already bypassed this unreviewed-opening branch.
        opening_text = ' '.join(normalize(p) for p in raw[start:]
                                if sha(normalize(p)) not in excluded_hashes)[:5000]
        modern_markers = list(re.finditer(
            r'\bСССР\b|\bсоветск[а-яё]*|\b(?:19[2-9]\d|20\d\d)\s*(?:г\.(?=\s|$)|год(?:а|у|ом|ы|ов|ах)?\b)',
            opening_text, re.I))
        if modern_markers:
            ledger.append({'action': 'queue_unbounded_opening_modern_context',
                           'block_index': start,
                           'evidence': [opening_text[max(0, m.start()-100):m.end()+160]
                                        for m in modern_markers[:8]]})
            return [], ledger, ['opening_modern_context_requires_review']
    ledger.append({'action': 'frontmatter_excluded', 'block_from': 0, 'block_to_exclusive': start})
    paragraphs = []
    for i, original in enumerate(raw[start:reviewed_end], start):
        if TAIL.fullmatch(original):
            ledger.append({'action': 'editorial_tail_excluded', 'block_index': i, 'heading': original})
            flags.append('editorial_tail_removed')
            break
        p = normalize(original)
        reasons = []
        if sha(p) in excluded_hashes:
            reasons.append('source_pinned_reviewed_paragraph_exclusion')
        short_bibliography = (len(p) < 650 and re.search(r'(?:Издательство|Москва|М\.:|Л\.:|Оттава|University|Museum)', p, re.I)
                              and re.search(r'\b(?:19[2-9]\d|20\d\d)\b', p))
        if META.search(p) or BIB.search(p) or short_bibliography:
            reasons.append('editorial_or_bibliographic')
        if len(p) < 650 and CREDIT.search(p):
            reasons.append('editorial_credit')
        if journal_footer(p, i, len(raw)):
            reasons.append('journal_publication_footer')
        if scholarly.search(p):
            reasons.append('scholarly_editorial_note')
        foreign_letters = len(re.findall('[їєґўђћљњџјЇЄҐЎЂЋЉЊЏЈ]', original))
        if foreign_letters >= 3:
            reasons.append('other_cyrillic_language_block')
        if re.search(r'\[[^\]]+\]|\{[^}]+\}|<[^>]*>|[<>]', p):
            reasons.append('unresolved_annotation')
        p, controls = repair_format_controls(p)
        p, repairs = repair_homoglyphs(p)
        problems = issues(p)
        if problems:
            reasons.append('unresolved_ocr_or_script')
        if re.search(r'[А-Яа-яЁё][,.;:!?]\d{1,3}(?:\s|$)', p):
            reasons.append('attached_note_number')
        cyr = len(CYR.findall(p)); latin = len(re.findall('[A-Za-z]', p))
        if latin > max(20, cyr * .15):
            reasons.append('foreign_language_block')
        if not cyr:
            # Separators are not removal gaps, but carry no training prose.
            ledger.append({'action': 'separator_skipped', 'block_index': i, 'text': p})
            continue
        output_index = len(paragraphs)
        ledger.append({'action': 'excluded' if reasons else 'retained', 'block_index': i,
                       'paragraph_index': output_index, 'reasons': reasons,
                       'before_sha256': sha(original), 'after_sha256': sha(p),
                       **({'issues': problems} if problems else {}),
                       **({'format_controls_removed': controls} if controls else {}),
                       **({'homoglyph_repairs': repairs} if repairs else {}),
                       **({'text': original} if reasons else {})})
        paragraphs.append(None if reasons else p)
    if any(p and len(p) > 12000 for p in paragraphs):
        flags.append('large_paragraph_review')
    return paragraphs, ledger, flags
