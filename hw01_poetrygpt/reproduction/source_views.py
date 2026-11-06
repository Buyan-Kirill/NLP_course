"""Deterministic decoded views for the frozen source-selection recipes.

Filtering, deduplication and train/validation/test decisions are frozen in the
selection manifests. These views recover their source spans; they never infer
missing words or silently accept a changed upstream edition.
"""
import csv
from functools import lru_cache
import io
from html.parser import HTMLParser
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
import sys

# Original source adapters also support direct execution from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scripts.build_prose_v5 import strip_html
from scripts.build_prose_v6 import normalize_source_paragraph, repair_format_controls, repair_homoglyphs
from scripts.extract_web_v7 import Blocks, normalize
from scripts.corpus_sources import clean_text
from scripts.collect_poetry_v7 import extract as lib_poetry

csv.field_size_limit(10_000_000)


@lru_cache(maxsize=2)
def structured_records(path, kind, field=None):
    """Parse each shared XML/CSV once, bounded to two documents in memory."""
    if kind == 'xml':
        return [clean_text(item.findtext('text', ''))
                for item in ET.parse(path).getroot()]
    if kind == 'csv':
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            return [clean_text(row[field]) for row in csv.DictReader(stream)]
    raise ValueError('Unsupported structured source: ' + kind)


def poetry_view(path, spec):
    kind = spec['kind']
    if kind in ('xml', 'csv'):
        return structured_records(str(path), kind, spec.get('field'))[spec['record'] - 1]
    if kind == 'zip':
        with zipfile.ZipFile(path) as archive:
            return clean_text(archive.read(spec['member']).decode('utf-8-sig'))
    if kind == 'lib_poetry':
        return lib_poetry(Path(path).read_bytes())[0]
    if kind in ('rvb_line', 'rvb_verse'):
        parser = RVBLineParser() if kind == 'rvb_line' else RVBVerseParser()
        parser.feed(Path(path).read_bytes().decode(spec['encoding']))
        return parser.text()
    raise ValueError('Unsupported poetry source: ' + kind)
def views(payload, kind, requested=None):
    if kind == 'RSD':
        text = unicodedata.normalize('NFC', payload.decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n'))
        out = []
        for p in text.splitlines():
            p = p.strip()
            p = re.sub(r'_([^_\n]+)_', r'\1', p)
            p = re.sub(r'(?<=[А-Яа-яЁё]),(?=[А-Яа-яЁё])', ', ', p)
            p = re.sub(r'(?<=[а-яё])([.!?;:])(?=[А-ЯЁ])', r'\1 ', p)
            p = re.sub(r'([А-Яа-яЁё][,.])(?=\d{1,2}-го (?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря))', r'\1 ', p)
            p = re.sub(r'\[\d+\]|\{\d+\}', '', p)
            p = re.sub(r'[ \t\u00a0]+', ' ', p)
            p = repair_homoglyphs(repair_format_controls(p)[0])[0]
            out.append(p)
        return {'rsd': '\n'.join(out)}
    declared = re.search(br'charset\s*=\s*["\x27]?([A-Za-z0-9_-]+)', payload[:5000], re.I)
    enc = {'windows-1251':'cp1251','cp1251':'cp1251','koi8-r':'koi8-r','utf-8':'utf-8-sig'}.get(declared[1].decode('ascii').lower() if declared else '')
    if enc is None:
        try:
            payload.decode('utf-8-sig'); enc='utf-8-sig'
        except UnicodeDecodeError:
            enc='cp1251'
    source=payload.decode(enc, errors='replace').replace('\r\r\n','\n').replace('\r\n','\n').replace('\r','\n')
    opening=re.search(r'<!--[^>]*Собственно произведение[^>]*-->',source,re.I)
    end=source.find('</noindex>',opening.end())
    body=source[opening.end():end]
    legacy = []
    if requested != 'lib_blocks':
        legacy = [normalize_source_paragraph(p) for p in strip_html(body).splitlines() if p.strip()]
        legacy = [repair_homoglyphs(repair_format_controls(p)[0])[0] for p in legacy]
    if requested == 'lib_legacy':
        return {'lib_legacy': '\n\n'.join(legacy)}
    body=re.sub(r'<a\b[^>]*href=["\']?#[^>]*>\s*(?:\d+|\[\d+\]|\*)\s*</a>', '', body, flags=re.I|re.S)
    parser=Blocks();parser.feed(body);parser.flush()
    modern=[repair_homoglyphs(repair_format_controls(normalize(p))[0])[0] for p in parser.blocks]
    return {'lib_legacy':'\n\n'.join(legacy), 'lib_blocks':'\n\n'.join(modern)}

class RVBLineParser(HTMLParser):
    """Pushkin edition has explicit line/linecont spans with nested references."""
    def __init__(self):
        super().__init__(convert_charrefs=True); self.span=0;self.active=None;self.sup=0;self.buffer=[];self.lines=[];self.cont=False
    def handle_starttag(self,tag,attrs):
        classes=set(dict(attrs).get('class','').split())
        if tag=='span':
            self.span+=1
            if classes.intersection({'line','linecont'}) and self.active is None:
                self.active=self.span;self.buffer=[];self.cont='linecont' in classes
        if tag=='sup':self.sup+=1
        if self.active is None and classes.intersection({'stanza','versepara','versepara2'}):self.lines.append('')
    def handle_endtag(self,tag):
        if tag=='sup':self.sup=max(0,self.sup-1)
        if tag=='span':
            if self.active==self.span:
                t=''.join(self.buffer).strip()
                if self.cont and self.lines:self.lines[-1]+=' '+t
                else:self.lines.append(t)
                self.active=None
            self.span-=1
    def handle_data(self,data):
        if self.active is not None and not self.sup:self.buffer.append(re.sub(r'\s+',' ',data))
    def text(self):return clean_text('\n'.join(self.lines))

class RVBVerseParser(HTMLParser):
    """Read flex-poem containers, including verse after malformed p endings."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth=0; self.active=None; self.skip=[]; self.parts=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs); classes=set(attrs.get('class','').split())
        if tag=='div':
            self.depth+=1
            if 'flex-poem' in classes:self.active=self.depth
        if self.active is None:return
        if tag in ('sup','script','h1','h2','h3','h4') or classes.intersection({'verseno','number','page','page-straight','epigraph-right','epigraph-author','flex-epigraph-text','date-poem','chapter-number','chapter-number-latin','footnote','dotted-spaced','asterisk-center'}):
            self.skip.append(tag)
        if not self.skip and tag in ('br','p','div'):self.parts.append('\n')
    def handle_endtag(self,tag):
        if self.skip and tag==self.skip[-1]:self.skip.pop()
        elif self.active is not None and not self.skip and tag in ('p','div'):self.parts.append('\n')
        if tag=='div':
            if self.active==self.depth:self.active=None
            self.depth-=1
    def handle_data(self,data):
        if self.active is not None and not self.skip:self.parts.append(re.sub(r'\s+',' ',data))
    def text(self):
        # Page boundaries cannot safely be inferred to be authorial stanzas.
        return clean_text('\n'.join(s.strip() for s in ''.join(self.parts).splitlines() if s.strip()))

