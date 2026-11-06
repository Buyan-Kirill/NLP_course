"""Whole-work serialization with the historical token and metadata contract."""
from collections import defaultdict
from itertools import islice
import json
import math
from pathlib import Path
import numpy as np
from poetrygpt.serialization import SPECIAL_TOKENS, special_ids
OUT = None
LINE_ENDINGS = {}
META_KEYS = ("family_id", "group_id", "author", "author_id", "genre", "genres",
             "original_work_id", "source_date_values", "source_metadata", "source_genre", "annotation_status",
             "topic", "topics", "themes", "source_themes", "theme_status",
             "dates", "date_basis", "date_evidence", "written_from", "written_to",
             "publication_year", "edition_year", "sources", "source_id", "source_url",
             "provenance", "release_provenance", "originality", "originality_basis",
             "orthography", "core_style", "core_style_basis", "quality", "form_evidence", "verse")

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.write_bytes(text.replace('\n', LINE_ENDINGS.get(path.name, '\n')).encode('utf8'))

def sidecar(row, metadata, domain):
    combined = {**metadata.get(row["work_id"], {}), **{key: row[key] for key in META_KEYS if key in row}}
    return {"domain": domain, **combined}

def strata_keys(metadata, content_tokens):
    keys = [f"domain:{metadata['domain']}", f"author:{metadata.get('author') or 'unknown'}",
            "token_length:" + ("<128" if content_tokens < 128 else "128-511" if content_tokens < 512 else ">=512")]
    for field in ("genre", "genres", "topic", "topics", "themes", "date_basis"):
        value = metadata.get(field)
        values = value if isinstance(value, list) else [value] if value is not None else []
        keys.extend(f"{field}:{item}" for item in values)
    for source in metadata.get("sources", []):
        label = source.get("dataset", source.get("source_id", "unknown")) if isinstance(source, dict) else source
        keys.append(f"source:{label}")
    dates = metadata.get("dates")
    written = dates.get("written_from") if isinstance(dates, dict) else metadata.get("written_from")
    keys.append(f"written_period:{written // 50 * 50}-{written // 50 * 50 + 49}" if isinstance(written, int) else "written_period:unknown")
    return sorted(set(keys))

def prepare_pretrain(model, rows, metadata, domain, split, out=OUT):
    if not rows:
        raise ValueError(f"Empty {domain}/{split}")
    ids_special = special_ids(model)
    raw_path = out / f".{domain}_{split}.uint16.tmp"
    array_path = out / f"{domain}_{split}.npy"
    offsets, cursor = [], 1
    strata = defaultdict(lambda: {"works": 0, "content_tokens": 0})
    seen = set()
    try:
        with raw_path.open("wb") as raw:
            np.asarray([ids_special["<|bos|>"]], dtype=np.uint16).tofile(raw)
            iterator = iter(rows)
            while batch := list(islice(iterator, 32)):
                texts = [row['text'] for row in batch]
                if any(not text.strip() or any(marker in text for marker in SPECIAL_TOKENS) for text in texts):
                    raise ValueError('Empty text or a literal special-token marker')
                # Batch independent complete works; never tokenize paragraphs
                # separately. Rust parallelism changes throughput, not BPE cuts.
                encoded = model.encode_batch(texts, add_special_tokens=False)
                contents = [item.ids for item in encoded]
                if model.decode_batch(contents, skip_special_tokens=False) != texts:
                    raise ValueError('Whole-work tokenizer roundtrip failed')
                for row, content in zip(batch, contents):
                    work_id = row['work_id']
                    if work_id in seen:
                        raise ValueError(f'Duplicate {domain}/{split} work: {work_id}')
                    seen.add(work_id)
                    np.asarray(content, dtype=np.uint16).tofile(raw)
                    np.asarray([ids_special['<|eos|>']], dtype=np.uint16).tofile(raw)
                    end = cursor + len(content) + 1
                    meta = sidecar(row, metadata, domain)
                    offsets.append({'work_id': work_id, 'start': cursor, 'end': end, **meta})
                    for key in strata_keys(meta, len(content)):
                        strata[key]['works'] += 1
                        strata[key]['content_tokens'] += len(content)
                    cursor = end
                if len(offsets) % 2048 == 0:
                    print(f'{domain}/{split}: {len(offsets)}/{len(rows)} works, {cursor-1:,} tokens', flush=True)
        source = np.memmap(raw_path, mode="r", dtype=np.uint16, shape=(cursor,))
        destination = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.uint16, shape=(cursor,))
        for start in range(0, cursor, 1_000_000):
            destination[start:start + 1_000_000] = source[start:start + 1_000_000]
        destination.flush()
        del source, destination
    finally:
        # This is one exact temporary file created above, never a recursive delete.
        if raw_path.exists():
            raw_path.unlink()
    restored = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if int(restored[0]) != ids_special["<|bos|>"] or len(restored) != cursor:
        raise ValueError("Saved stream boundary/length mismatch")
    endings = np.asarray([row['end'] - 1 for row in offsets], dtype=np.int64)
    if not np.all(restored[endings] == ids_special['<|eos|>']):
        raise ValueError('Missing work EOS')
    # The caller verifies the entire NPY and offsets against frozen SHA256.
    # Causal shift and tail padding are checked by contract tests, rather than
    # constructing 210M Python labels a second time during data preparation.
    targets = cursor - 1
    count = math.ceil(targets / 511)
    last_positions = cursor - (count - 1) * 511
    write_json(out / f"{domain}_{split}_offsets.json", offsets)
    return {"works": len(rows), "stream_tokens": cursor, "content_tokens": cursor - len(rows) - 1,
            "causal_targets": targets, "windows": count, "last_window_positions": last_positions,
            "roundtrip_works": len(rows), "dtype": "uint16", "dropped_or_truncated": 0}, dict(sorted(strata.items()))
