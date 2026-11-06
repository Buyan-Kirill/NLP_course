"""Download checksum-pinned resources; never execute an installer or unpickle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import ssl
import time
import uuid
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def https_context():
    """Keep system trust and add Mozilla roots where Python misses an issuer."""
    context = ssl.create_default_context()
    try:
        import certifi
    except ModuleNotFoundError:
        return context
    context.load_verify_locations(certifi.where())
    return context


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(url, destination, expected, size=None, *, equivalent=None):
    """Reuse verified cache entries; publish a download only after verification."""
    destination = Path(destination)
    if destination.exists():
        if sha256(destination) != expected and not (equivalent and equivalent(destination)):
            raise ValueError(f'Cached file differs from SHA256: {destination}')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        temporary = None
        try:
            request = Request(url, headers={'User-Agent': 'PoetryGPT-reproduction/1.0'})
            temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.partial')
            with urlopen(request, timeout=90, context=https_context()) as response, temporary.open('xb') as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            matches = ((size is None or temporary.stat().st_size == size)
                       and sha256(temporary) == expected)
            if not matches and not (equivalent and equivalent(temporary)):
                raise ValueError(f'Upstream bytes changed; expected {expected}: {url}')
            os.replace(temporary, destination)
            return destination
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise AssertionError('Download retry loop did not finish')


def extract_checked(archive_path, root, files):
    """Extract only declared regular files, with no path escapes or clobbering."""
    root = Path(root).resolve()
    declared = {item['path']: item for item in files}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(declared):
            raise ValueError('Archive entries differ from the resource manifest')
        # Validate the whole archive before publishing even its first entry.
        for name in names:
            info = archive.getinfo(name)
            target = (root / name).resolve()
            mode = info.external_attr >> 16
            if ('\\' in name or not target.is_relative_to(root) or target == root
                    or stat.S_ISLNK(mode) or info.is_dir()):
                raise ValueError('Unsafe archive member: ' + name)
            item = declared[name]
            if info.file_size != item['bytes']:
                raise ValueError('Archive member size mismatch: ' + name)
            with archive.open(name) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != item['sha256']:
                raise ValueError('Archive member checksum mismatch: ' + name)
            if target.exists() and sha256(target) != digest:
                raise FileExistsError('Preserve existing local file: ' + str(target))
        for name in names:
            target = (root / name).resolve()
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open('xb') as destination:
                shutil.copyfileobj(source, destination)


def inference():
    release = json.loads((ROOT / 'reproduction/resources.json').read_text('utf8'))
    item = release['inference']
    if not (ROOT / '.cache' / item['name']).exists():
        print(f'Загружаю ресурсы: {item["bytes"]/1_000_000:.0f} МБ…', flush=True)
    archive = fetch(item['url'], ROOT / '.cache' / item['name'], item['sha256'], item['bytes'])
    extract_checked(archive, ROOT, item['files'])
    return {'resource': 'inference', 'verified_files': len(item['files'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('resource', choices=['inference'])
    parser.parse_args()
    print(json.dumps(inference(), ensure_ascii=False))


if __name__ == '__main__':
    main()
