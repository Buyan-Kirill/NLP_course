"""Read downloaded snapshots; emit audit metrics without building a training set."""

import csv
import hashlib
import html
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "audit" / "sources"
csv.field_size_limit(10_000_000)


def csv_rows(path, delimiter=","):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def normalize(text):
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    text = unicodedata.normalize("NFC", text).casefold().replace("ё", "е")
    return " ".join(re.findall(r"\w+", text))


def digest(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def profile(rows):
    texts = [r["text"] for r in rows]
    keys = [digest(t) for t in texts if normalize(t)]
    return {
        "rows": len(rows),
        "text_utf8_bytes": sum(len(t.encode("utf-8")) for t in texts),
        "characters": sum(map(len, texts)),
        "empty_texts": sum(not t.strip() for t in texts),
        "unique_raw_texts": len(set(texts)),
        "unique_normalized_nonempty_texts": len(set(keys)),
        "duplicate_normalized_excess_rows": len(keys) - len(set(keys)),
        "top_authors": Counter(r.get("author", "") for r in rows).most_common(12),
    }


def in_period(row, start=1800, end=1850):
    a, b = row.get("date_from", ""), row.get("date_to", "")
    return a.isdigit() and b.isdigit() and start <= int(a) <= int(b) <= end


def archive_rows(key, archive, metadata_path, delimiter, author_key):
    metadata = csv_rows(CACHE / key / metadata_path, delimiter)
    by_id = {r["id"]: r for r in metadata}
    rows = []
    with zipfile.ZipFile(CACHE / key / archive) as z:
        for path in sorted(z.namelist()):
            if not path.endswith(".txt") or path.startswith("__MACOSX"):
                continue
            record_id = Path(path).name.split("_", 1)[0]
            row = dict(by_id.get(record_id, {}))
            row.update(id=record_id, archive_path=path, text=z.read(path).decode("utf-8-sig"), author=row.get(author_key, ""))
            rows.append(row)
    return rows, metadata


def main():
    poetry = [{child.tag: child.text or "" for child in item} for item in ET.parse(CACHE / "poetry/datasets/corpus/all.xml").getroot()]
    ru31 = csv_rows(CACHE / "ru31/datasets/ru_poems_31k_v1.0.0.csv")
    result = {"poetry": profile(poetry), "ru31": profile(ru31)}
    result["poetry"]["missing_date_from"] = sum(not r["date_from"] for r in poetry)
    result["poetry"]["missing_date_to"] = sum(not r["date_to"] for r in poetry)
    result["poetry"]["invalid_intervals"] = sum(r["date_from"].isdigit() and r["date_to"].isdigit() and int(r["date_from"]) > int(r["date_to"]) for r in poetry)
    result["poetry"]["period_1800_1850"] = profile([r for r in poetry if in_period(r)])
    result["poetry"]["period_1780_1870"] = profile([r for r in poetry if in_period(r, 1780, 1870)])
    sets = {"poetry": {digest(r["text"]) for r in poetry}, "ru31": {digest(r["text"]) for r in ru31}}
    result["overlap"] = {"poetry_ru31_normalized_unique_texts": len(sets["poetry"] & sets["ru31"])}
    metadata = csv_rows(CACHE / "c1835/corpus_1835/csv/texts_metadata.csv")
    by_id = {r["text_id"]: r for r in metadata}
    authors = {r["A_ID"]: r for r in csv_rows(CACHE / "c1835/corpus_1835/csv/authors.csv")}
    sources = {r["source_id"]: r for r in csv_rows(CACHE / "c1835/corpus_1835/csv/sources.csv")}
    c1835 = []
    original_texts = csv_rows(CACHE / "c1835/corpus_1835/csv/texts.csv")
    for row in original_texts:
        meta = by_id[row["text_id"]]
        c1835.append({**meta, "text": row["text_cln"], "author": authors.get(meta["A_ID"], {}).get("author_name", ""), "publication_year": sources.get(meta["source_id"], {}).get("year", "")})
    elegies, elegy_meta = archive_rows("elegies", "corpus.zip", "elegies_metadata.tsv", "\t", "Author")
    songs, song_meta = archive_rows("songs", "RP_corpus/corpus_split.zip", "RP_corpus/RP_data.csv", ";", "Name")
    corpora = {"poetry": poetry, "ru31": ru31, "c1835": c1835, "elegies": elegies, "songs": songs}
    for key in ["c1835", "elegies", "songs"]:
        result[key] = profile(corpora[key])
        sets[key] = {digest(r["text"]) for r in corpora[key] if normalize(r["text"])}
    result["c1835"].update(
        text_fields_utf8_bytes={field: sum(len(r[field].encode("utf-8")) for r in original_texts) for field in ["text_raw", "text_cln", "text_lemm", "text_acc"]},
        publication_year_counts=dict(Counter(r["publication_year"] for r in c1835)),
        missing_author_ids=sum(not r["A_ID"] for r in c1835),
        missing_author_names=sum(not r["author"] for r in c1835),
        unmatched_source_ids=sum(r["source_id"] not in sources for r in c1835),
        unmatched_author_ids=sum(bool(r["A_ID"]) and r["A_ID"] not in authors for r in c1835),
        distinct_text_ids=len({r["text_id"] for r in c1835}),
    )
    for key, meta, rows in [("elegies", elegy_meta, elegies), ("songs", song_meta, songs)]:
        text_ids = {r["id"] for r in rows}
        meta_ids = {r["id"] for r in meta}
        result[key].update(metadata_rows=len(meta), distinct_text_ids=len(text_ids), metadata_without_text=len(meta_ids - text_ids), texts_without_metadata=len(text_ids - meta_ids), year_counts=dict(Counter(r.get("Year", "") for r in rows)))
        result[key]["dates_outside_1800_1850_or_nonnumeric"] = [{"id":r["id"], "year":r.get("Year", ""), "author":r["author"]} for r in rows if not r.get("Year", "").isdigit() or not 1800 <= int(r["Year"]) <= 1850]
    for i, key in enumerate(corpora):
        for other in list(corpora)[i+1:]:
            result["overlap"][f"{key}_{other}_normalized_unique_texts"] = len(sets[key] & sets[other])
    result["all_five_union_normalized_unique_texts"] = len(set.union(*sets.values()))
    result["songs"]["available_type_counts"] = dict(Counter(r.get("Type", "") for r in songs))
    result["songs"]["metadata_type_counts"] = dict(Counter(r.get("Type", "") for r in song_meta))
    selected = [r for r in poetry if in_period(r)] + c1835 + [r for r in elegies + songs if r.get("Year", "").isdigit() and 1800 <= int(r["Year"]) <= 1850]
    unique = {digest(r["text"]): r for r in selected}
    result["provisional_dated_pool"] = {"note": "Mixed date semantics: PoetryCorpus interval, c1835 source publication year, elegies/songs Year. Not a finalized training corpus.", "rows_before_dedup": len(selected), "unique_normalized_texts": len(unique), "representative_text_utf8_bytes": sum(len(r["text"].encode("utf-8")) for r in unique.values())}
    result["examples"] = {}
    for key, rows in [("poetry", poetry), ("ru31", ru31)]:
        selected = next((r for r in rows if "Пушкин" in r["author"] and "Любви, надежды" in r["text"]), rows[0])
        result["examples"][key] = {k: v for k, v in selected.items() if k != "text"}
        result["examples"][key]["excerpt"] = "\n".join(selected["text"].splitlines()[:4])
    for key, rows, predicate in [
        ("c1835", c1835, lambda r: "Пушкин" in r["author"] and r["text_title"] == "Туча"),
        ("elegies", elegies, lambda r: r["id"] == "334"),
        ("songs", songs, lambda r: r["id"] == "110"),
    ]:
        row = next((r for r in rows if predicate(r)), rows[0])
        result["examples"][key] = {k:v for k,v in row.items() if k != "text"}
        result["examples"][key]["excerpt"] = "\n".join([line for line in row["text"].splitlines() if line.strip()][:4])
    destination = ROOT / "audit" / "profile.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
