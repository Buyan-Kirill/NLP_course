"""Load pinned RPST assets with platform-native dependencies outside Windows."""
import hashlib
import importlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts/evaluators/rpst'


def verify_assets(artifacts):
    manifest = json.loads((artifacts / 'version.json').read_text('utf8'))
    for item in manifest['verified_files']:
        path = (artifacts / item['path']).resolve()
        if not path.is_relative_to(artifacts.resolve()):
            raise ValueError('RPST manifest path escapes its artifact directory')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != item['size_bytes'] or digest != item['sha256']:
            raise ValueError('RPST artifact changed: ' + item['path'])
    return manifest


def source_paths(artifacts, manifest):
    """Never add the bundled Windows binary directory to a Unix import path."""
    paths = []
    for value in manifest['import_paths']:
        path = (artifacts / value).resolve()
        if not path.is_relative_to(artifacts.resolve()):
            raise ValueError('RPST import path escapes its artifact directory')
        if Path(value).parts[0] == 'source':
            paths.append(str(path))
    if len(paths) != 2:
        raise ValueError('Expected pinned RPST and rusyllab source directories')
    return paths


def load_native_tool(artifacts=ARTIFACTS):
    """Verify data before unpickling; use the interpreter's native extensions."""
    manifest = verify_assets(artifacts)
    expected = {r['name']: r['version'] for r in manifest['target_dependencies']}
    versions = {}
    for name, wanted in expected.items():
        found = package_version(name)
        if found != wanted:
            raise RuntimeError(f'{name}=={wanted} required; found {found}. Install requirements.txt.')
        versions[name] = found
    for name in ('torch', 'numpy', 'huggingface_hub', 'PyYAML'):
        versions[name] = package_version(name)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    sys.dont_write_bytecode = True
    sys.path[:0] = source_paths(artifacts, manifest)
    origins = {}
    for name in ('ufal.udpipe', 'jellyfish'):
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if origin.is_relative_to((artifacts / 'dependencies').resolve()):
            raise RuntimeError('Platform-native package required: ' + name)
        binaries = sorted(origin.parent.glob('*.so')) + sorted(origin.parent.glob('*.pyd'))
        checked = []
        for path in [origin, *binaries]:
            with path.open('rb') as stream:
                checked.append(dict(path=str(path), sha256=hashlib.file_digest(stream, 'sha256').hexdigest()))
        origins[name] = checked
    import torch
    import russian_scansion
    torch.set_num_threads(2)
    with torch.inference_mode():
        tool = russian_scansion.create_rpst_instance(models_dir=str(artifacts / 'models'))
    devices = {str(p.device) for p in tool.accentuator.stress_model.model.parameters()}
    if devices != {'cpu'}:
        raise RuntimeError('RPST must run on CPU')
    manifest = dict(manifest, native_host=dict(
        platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
        packages=versions, native_imports=origins,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return tool, torch, manifest
