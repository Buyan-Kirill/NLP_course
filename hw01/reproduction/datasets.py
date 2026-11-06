"""Recover the selected corpus from public sources and frozen span recipes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import gzip
import json
from pathlib import Path
import shutil

from reproduction.download import ROOT, fetch, sha256

RECIPES = ROOT / 'reproduction/corpus'
CACHE = ROOT / '.cache/sources'


def read_gzip(name):
    with gzip.open(RECIPES / name, 'rt', encoding='utf8') as stream:
        return json.load(stream)


def catalog():
    return json.loads((RECIPES / 'manifest.json').read_text('utf8'))


def source_path(identifier, sources):
    item = sources[identifier]
    bodies = json.dumps(item.get('body_views', []), sort_keys=True)
    return verified_source(identifier, item['url'], item['sha256'], item['bytes'], bodies)


@lru_cache(maxsize=None)
def verified_source(identifier, url, expected, size, bodies):
    # Shared XML/CSV files contain thousands of records: hash once per process,
    # then verify every reconstructed text and the final serialized output.
    body_specs = json.loads(bodies)

    def equivalent(path):
        # A library page may change its counters or footer. Accept changed
        # transport bytes only when every frozen decoded body view is identical.
        import hashlib
        from reproduction.source_views import views, poetry_view
        for spec in body_specs:
            text = (views(path.read_bytes(), 'Lib.ru_Classic', requested=spec['view'])[spec['view']]
                    if spec['domain'] == 'prose' else poetry_view(path, spec['view']))
            if hashlib.sha256(text.encode('utf8')).hexdigest() != spec['sha256']:
                return False
        receipt = CACHE / (identifier + '.transport.json')
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(dict(url=url, expected_raw_sha256=expected,
            observed_raw_sha256=sha256(path), decoded_body_sha256_verified=True), indent=2)+'\n', 'utf8')
        return True

    return fetch(url, CACHE / (identifier + '.raw'), expected, size,
                 equivalent=equivalent if body_specs else None)


def download(domain='all', workers=2):
    manifest = catalog()
    identifiers = set()
    for name in ('prose', 'poetry') if domain == 'all' else (domain,):
        identifiers.update(manifest['domains'][name]['sources'])
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(source_path, key, manifest['sources']): key
                for key in sorted(identifiers)}
        for number, future in enumerate(as_completed(jobs), 1):
            key = jobs[future]
            try:
                future.result()
            except (OSError, ValueError) as error:
                failures.append(dict(source=key, url=manifest['sources'][key]['url'], error=str(error)))
            if number % 25 == 0 or number == len(jobs):
                print(f'Sources: {number}/{len(jobs)}, errors: {len(failures)}', flush=True)
    if failures:
        path = ROOT / '.cache/source_failures.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + '\n', 'utf8')
        raise RuntimeError(f'{len(failures)} sources unavailable or changed; see {path}')
    return len(identifiers)


def select_text(text, selection):
    """Apply source intervals and short, reviewed punctuation/OCR corrections."""
    def span(pair):
        start, stop = pair
        if not (type(start) is int and type(stop) is int and 0 <= start <= stop <= len(text)):
            raise ValueError('Selection interval outside its pinned source')
        return text[start:stop]

    if 'spans' in selection:
        return selection['separator'].join(span(pair) for pair in selection['spans'])
    return ''.join(piece if isinstance(piece, str) else span(piece)
                   for piece in selection['pieces'])


def restore_sidecar(relative, expected, compressed):
    target = ROOT / relative
    if target.exists():
        if sha256(target) != expected:
            raise FileExistsError('Existing metadata differs: ' + relative)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.partial')
    with gzip.open(RECIPES / compressed, 'rb') as source, temporary.open('wb') as destination:
        shutil.copyfileobj(source, destination)
    if sha256(temporary) != expected:
        temporary.unlink()
        raise ValueError('Restored metadata differs: ' + relative)
    temporary.replace(target)


def build(domain='all'):
    from reproduction.source_views import views, poetry_view
    manifest = catalog()
    for name in ('prose', 'poetry') if domain == 'all' else (domain,):
        spec = manifest['domains'][name]
        recipes = read_gzip(spec['recipes'])
        by_id = {row['metadata']['work_id']: row for row in recipes}

        @lru_cache(maxsize=2)
        def prose_views(identifier, view):
            item = manifest['sources'][identifier]
            path = source_path(identifier, manifest['sources'])
            return views(path.read_bytes(), item['dataset'], requested=view)

        def render(recipe):
            selection = recipe['selection']
            if name == 'prose':
                text = prose_views(selection['source'], selection['view'])[selection['view']]
            else:
                path = source_path(selection['source'], manifest['sources'])
                text = poetry_view(path, selection['view'])
            text = select_text(text, selection)
            import hashlib
            expected = recipe['text_sha256']
            if hashlib.sha256(text.encode('utf8')).hexdigest() != expected:
                raise ValueError('Reconstructed text differs: ' + recipe['metadata']['work_id'])
            row = dict(recipe['metadata'])
            row['text'] = text  # text key keeps its original position in the row
            return row

        for filename, output in spec['outputs'].items():
            target = ROOT / filename
            if target.exists():
                if sha256(target) != output['sha256']:
                    raise FileExistsError('Existing corpus differs: ' + filename)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.jsonl.partial')
            with temporary.open('w', encoding='utf8', newline='') as stream:
                for index, work_id in enumerate(output['ids']):
                    row = render(by_id[work_id])
                    if output.get('fields') is not None:
                        row = {key: row[key] for key in output['fields']}
                    ending = output['newline'] if output['terminal_newline'] or index < len(output['ids'])-1 else ''
                    stream.write(json.dumps(row, ensure_ascii=False) + ending)
                    if (index + 1) % 2000 == 0:
                        print(f'{filename}: {index + 1}/{len(output["ids"])}', flush=True)
            if sha256(temporary) != output['sha256']:
                raise ValueError('Rebuilt JSONL differs; preserve diagnostic partial: ' + filename)
            temporary.replace(target)
        for relative, sidecar in spec['sidecars'].items():
            restore_sidecar(relative, sidecar['sha256'], sidecar['compressed'])
        print(f'{name}: all reconstructed text and metadata hashes match', flush=True)


def prepare():
    from reproduction.prepare import prepare as serialize
    for domain, folder in [('prose', 'expanded'), ('poetry', 'poetry_supplement')]:
        serialize(domain, ROOT / 'data/prepared_v7' / folder)
    from reproduction.prepare_auxiliary import prepare as auxiliary
    auxiliary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['download', 'build', 'prepare'])
    parser.add_argument('--domain', choices=['all', 'prose', 'poetry'], default='all')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    if args.action == 'download':
        download(args.domain, args.workers)
    elif args.action == 'build':
        build(args.domain)
    else:
        if args.domain != 'all':
            parser.error('prepare verifies both domains and the auxiliary recipe')
        prepare()


if __name__ == '__main__':
    main()
