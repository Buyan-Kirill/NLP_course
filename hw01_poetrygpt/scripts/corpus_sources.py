"""Adapters for the inspected, pinned source snapshots; no network access."""

import hashlib
import html
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from profile_sources import CACHE, ROOT, archive_rows, csv_rows, normalize as base_normalize

MISSING = {"", "na", "nan", "null", "?", "[без подписи]", "без подписи", "б/п"}
REPOSITORIES = {
    "poetry": "IlyaGusev/PoetryCorpus", "ru31": "kskadart/ru-poetry-dataset",
    "c1835": "tonyamart/corpus_1835", "elegies": "tonyamart/rus_elegies",
    "songs": "perechen/russian.songs.fin.thesis",
}
PRIORITY = {"wikisource": 0, "rvb": 0, "c1835": 1, "elegies": 2, "poetry": 3, "ru31": 4, "songs": 5}
HOMOGLYPHS = str.maketrans("ABEKMHOPCTXYaeopcyx", "АВЕКМНОРСТХУаеорсух")


def normalize(text):
    # Comparison only: original typography/spelling is retained in training text.
    def repair_match(match):
        token=match.group()
        if re.search(r"[А-Яа-яЁё]",token) and re.search(r"[A-Za-z]",token):
            return token.translate(HOMOGLYPHS)
        return token
    return base_normalize(re.sub(r"[A-Za-zА-Яа-яЁё]+",repair_match,text))


def value(x):
    x = str(x or "").strip()
    return None if x.casefold() in MISSING else x


def year(x):
    x = str(x or "").strip()
    return int(x) if re.fullmatch(r"1[0-9]{3}", x) else None


def clean_text(text):
    text = unicodedata.normalize("NFC", html.unescape(text))
    text = re.sub(r"\r+\n?", "\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</?(?:i|b|em|strong|span|p|div|a|sup|sub|font|sic)(?:\s[^>]*)?>", "", text, flags=re.I)
    text = text.replace("\xa0", " ").replace("\u202f", " ").replace("\ufeff", "").replace("\u200b", "").replace("\xad", "")
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c) != "Cc")
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")
    return re.sub(r"\n{3,}", "\n\n", text)


class Authors:
    def __init__(self, author_rows, ru_rows):
        self.aliases = defaultdict(set)
        for row in author_rows:
            full = value(row["author_full_name"])
            if full:
                self.add(full, full)
                self.add(row["author_name"], full)
        # Only complete, unambiguous three-part names can supply extra aliases.
        for row in ru_rows:
            full = row["author"]
            parts = full.split()
            if len(parts) == 3 and all(re.fullmatch(r"[А-ЯЁа-яё-]+", p) for p in parts):
                existing = self.aliases.get(normalize(full))
                canonical = next(iter(existing)) if existing and len(existing) == 1 else full
                self.add(full, canonical)

    def add(self, alias, canonical):
        self.aliases[normalize(alias)].add(canonical)
        parts = canonical.split()
        if len(parts) == 3 and all(re.fullmatch(r"[А-ЯЁа-яё-]+", p) for p in parts):
            surname, first, patronymic = parts
            for alias in [f"{surname} {first[0]} {patronymic[0]}", f"{first} {patronymic} {surname}", f"{first} {surname}"]:
                self.aliases[normalize(alias)].add(canonical)

    def resolve(self, raw):
        raw = value(raw)
        if raw is None:
            return None
        candidates = self.aliases.get(normalize(raw), set())
        return next(iter(candidates)) if len(candidates) == 1 else raw


def registry():
    licenses = {"poetry": "Apache-2.0 (repository; underlying texts need separate attribution)", "ru31": "GNU FDL 1.3 + Apache-2.0 (README; per-row origin not supplied)", "c1835": "CC BY-NC-SA 4.0", "elegies": "CC BY-NC 4.0", "songs": None}
    downloads = [json.loads(line) for line in (CACHE / "downloads.jsonl").read_text(encoding="utf-8").splitlines()]
    result = {}
    for key, repo in REPOSITORIES.items():
        tree = json.loads((CACHE / key / "tree.json").read_text(encoding="utf-8"))
        files = {r["path"]: r for r in downloads if r["repository"] == repo}
        for path, entry in files.items():
            payload = (CACHE / key / path).read_bytes()
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"Source snapshot changed: {key}/{path}")
        result[key] = {"repository": repo, "revision": tree["sha"], "revision_kind": "git_tree", "license": licenses[key], "files": list(files.values()), "role": "deferred" if key == "songs" else "core" if key in {"poetry", "c1835", "elegies"} else "enrichment"}
    external = ROOT / "audit" / "web_records.json"
    if external.exists():
        rows = json.loads(external.read_text(encoding="utf-8"))
        for dataset in {r["dataset"] for r in rows}:
            pages = [r["source"] for r in rows if r["dataset"] == dataset]
            for source in pages:
                key = next(r["id"].split(":", 1)[1] for r in rows if r["source"] == source)
                payload = (ROOT / "audit" / "web" / f"{key}.html").read_bytes()
                if hashlib.sha256(payload).hexdigest() != source["metadata"]["html_sha256"]:
                    raise ValueError(f"Web snapshot changed: {key}")
            result[dataset] = {"role":"targeted_enrichment", "pages":pages, "license":"Work labelled PD-old-70; site CC BY-SA" if dataset=="wikisource" else "Electronic edition copyright notice; no explicit open dataset license on inspected pages"}
    return result


def load_records(reg):
    ru = csv_rows(CACHE / "ru31/datasets/ru_poems_31k_v1.0.0.csv")
    authors_raw = csv_rows(CACHE / "c1835/corpus_1835/csv/authors.csv")
    authors = Authors(authors_raw, ru)
    # Reviewed spelling/name variants observed in exact-text matches.
    for alias, canonical in {
        "Гаврила Державин": "Державин Гавриил Романович",
        "Апполон Григорьев": "Григорьев Аполлон Александрович",
        "Александр Бестужев": "Бестужев-Марлинский Александр Александрович",
        "Бестужев Александр Александрович": "Бестужев-Марлинский Александр Александрович",
        "Бестужев А. А.": "Бестужев-Марлинский Александр Александрович",
    }.items():
        authors.aliases[normalize(alias)] = {canonical}
    records = []

    def add(dataset, record_id, raw_text, author, title, path, evidence, **metadata):
        text = clean_text(raw_text)
        source = {"dataset": dataset, "record_id": record_id, "url": f"https://github.com/{reg[dataset]['repository']}/blob/{reg[dataset]['revision']}/{path}", "edition_year": metadata.pop("edition_year", None), "selected_text": False, "metadata": {"author": value(author), "title": value(title), **metadata}}
        records.append({"id": f"{dataset}:{record_id}", "dataset": dataset, "author": authors.resolve(author), "title": value(title), "text": text, "source": source, "period_evidence": evidence, "verse": metadata.get("verse"), "changed_by_cleanup": text != raw_text, "raw_bytes": len(raw_text.encode("utf-8"))})

    for index, item in enumerate(ET.parse(CACHE / "poetry/datasets/corpus/all.xml").getroot(), 1):
        row = {child.tag: child.text or "" for child in item}
        a, b = year(row.get("date_from")), year(row.get("date_to"))
        add("poetry", f"xml-item-{index:05d}", row["text"], row["author"], row["name"], "datasets/corpus/all.xml", {"kind": "source_date_interval", "from": a, "to": b}, date_from=row["date_from"], date_to=row["date_to"], themes=[x.text for x in item.findall("themes/value")], stanza_boundaries="removed_upstream")
    by_author = {r["A_ID"]: r for r in authors_raw}
    metas = csv_rows(CACHE / "c1835/corpus_1835/csv/texts_metadata.csv")
    by_id = {r["text_id"]: r for r in metas}
    editions = {r["source_id"]: r for r in csv_rows(CACHE / "c1835/corpus_1835/csv/sources.csv")}
    assert len(by_id) == len(metas), "Nonunique c1835 text IDs"
    for row in csv_rows(CACHE / "c1835/corpus_1835/csv/texts.csv"):
        meta = by_id[row["text_id"]]
        author = by_author.get(meta["A_ID"], {})
        edition = editions[meta["source_id"]]
        y = year(edition["year"])
        add("c1835", row["text_id"], row["text_cln"], value(author.get("author_full_name")) or author.get("author_name"), meta["text_title"], "corpus_1835/csv/texts.csv", {"kind": "historical_edition", "from": y, "to": y}, edition_year=y, source_id=meta["source_id"], author_id=value(meta["A_ID"]), bibliography=edition, verse={"meter": value(meta["meter"]), "feet": value(meta["feet"])}, expected_lines=meta["n_lines"], subtitle=value(meta["text_subtitle"]), text_field="text_cln")
    for dataset, archive, meta_path, sep, author_key in [
        ("elegies", "corpus.zip", "elegies_metadata.tsv", "\t", "Author"),
        ("songs", "RP_corpus/corpus_split.zip", "RP_corpus/RP_data.csv", ";", "Name"),
    ]:
        rows, _ = archive_rows(dataset, archive, meta_path, sep, author_key)
        for row in rows:
            y = year(row.get("Year"))
            add(dataset, row["id"], row["text"], row.get(author_key), row.get("Title"), archive, {"kind": "edition_year" if dataset == "elegies" else "source_date", "from": y, "to": y}, edition_year=y if dataset == "elegies" else None, archive_path=row["archive_path"], year_raw=row.get("Year"), bibliography={k:v for k,v in row.items() if k not in {"text", "archive_path", "author"}}, verse={"meter": value(row.get("Meter")), "feet": value(row.get("Стоп")) if dataset == "elegies" else None})
    for index, row in enumerate(ru, 1):
        add("ru31", f"csv-row-{index:05d}", row["text"], row["author"], row["poem_name"], "datasets/ru_poems_31k_v1.0.0.csv", {"kind": "unknown", "from": None, "to": None})
    external = ROOT / "audit" / "web_records.json"
    if external.exists():
        for row in json.loads(external.read_text(encoding="utf-8")):
            row["text"] = clean_text(row["text"])
            row["author"] = authors.resolve(row["author"])
            row["changed_by_cleanup"] = False
            row["raw_bytes"] = len(row["text"].encode("utf-8"))
            records.append(row)
    return records


def period_status(record):
    if record["dataset"] == "songs":
        return "deferred_source"
    evidence = record["period_evidence"]
    a, b = evidence["from"], evidence["to"]
    if a is None or b is None:
        return "unknown_date"
    if a > b:
        return "invalid_date"
    if b < 1800 or a > 1850:
        return "outside_period"
    if a < 1800 or b > 1850:
        return "boundary_date"
    return "eligible"
