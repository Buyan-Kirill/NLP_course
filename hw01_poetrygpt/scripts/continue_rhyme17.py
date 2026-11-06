"""Continue user poetry with the pinned system; retain every input and report."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path('configs/inference.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(path):
    path = (ROOT / path).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError('A project-local path is required')
    return path


def build_visible_input(text, tokenizer, tokenizer_sha256):
    """Pure serialization: no files, subprocesses, model, or phonetic imports."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Nonempty poetry text is required')
    if re.search(r'<\|[^\r\n]*?\|>', text):
        raise ValueError('Special-token marker strings are not allowed')
    normalized = text.replace('\r\n', '\n').replace('\r', '\n').rstrip('\n')
    first = [line for line in normalized.split('\n') if line.strip()][-1]
    context = '<|eos|>' + normalized + '\n'
    ids = tokenizer.encode(context, add_special_tokens=False).ids
    if tokenizer.decode(ids, skip_special_tokens=False) != context:
        raise ValueError('Tokenizer cannot preserve the canonical input exactly')
    case_id = 'user_' + hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    case = dict(id=case_id, group_id=case_id, first_line=first,
                context_text=context, context_ids=ids)
    return dict(schema='rhyme17-visible-runtime-input-1', split='inference',
                test_used=False, tokenizer_sha256=tokenizer_sha256, cases=[case])


def write_new(path, payload):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def run(text, config=DEFAULT_CONFIG, device='cuda'):
    if device not in ('cpu', 'cuda'):
        raise ValueError('Device must be cpu or cuda')
    config = local(config)
    config_sha = digest(config)
    cfg = json.loads(config.read_text(encoding='utf-8'))
    if digest(config) != config_sha:
        raise ValueError('Pinned config changed during read')
    expected_runtime = dict(config_sha256=config_sha,
                            **{key: dict(cfg[key]) for key in ('checkpoint', 'tokenizer', 'vocabulary')})
    token_ref = cfg['tokenizer']
    token_path = local(token_ref['path'])
    if digest(token_path) != token_ref['sha256']:
        raise ValueError('Pinned tokenizer SHA mismatch')
    from tokenizers import Tokenizer
    payload = build_visible_input(text, Tokenizer.from_file(str(token_path)), token_ref['sha256'])
    folder = local(Path('runs/inference17') / uuid.uuid4().hex)
    folder.mkdir(parents=True, exist_ok=False)
    visible, pool, report = (folder / name for name in ('input.json', 'pool.json', 'report.json'))
    write_new(folder / 'request.json', dict(text=text, config=str(config.relative_to(ROOT)),
                                          config_sha256=config_sha, device=device, remote=False))
    write_new(visible, payload)
    # RPST may hide CUDA in its process. Neither child may alter its sibling's env.
    parent_env = os.environ.copy()
    def child(script, arguments, log_name, *, module=False):
        entry = ['-m', script] if module else [str(ROOT / 'scripts' / script)]
        command = [sys.executable, '-B', '-X', 'utf8', *entry, *map(str, arguments)]
        with (folder / log_name).open('x', encoding='utf-8') as log:
            subprocess.run(command, cwd=ROOT, env=parent_env.copy(), shell=False,
                           check=True, stdout=log, stderr=subprocess.STDOUT)
    pool_builder = cfg.get('pool_builder', 'scripts/prepare_rhyme17_runtime_pool.py')
    if pool_builder not in ('scripts/prepare_rhyme17_runtime_pool.py', 'scripts/prepare_rhyme22_runtime_pool.py'):
        raise ValueError('Unsupported pool builder in runtime config')
    builder_sha = cfg.get('pool_builder_sha256')
    if pool_builder != 'scripts/prepare_rhyme17_runtime_pool.py' and digest(local(pool_builder)) != builder_sha:
        raise ValueError('Pinned pool builder SHA mismatch')
    child('reproduction.prepare_pool', ['--builder', pool_builder, '--input', visible,
          '--output', pool], 'pool.log', module=True)
    if builder_sha is not None and digest(local(pool_builder)) != builder_sha:
        raise ValueError('Pool builder changed during preparation')
    if digest(config) != config_sha or digest(token_path) != token_ref['sha256']:
        raise ValueError('Pinned config/tokenizer changed during preparation')
    child('reproduction.local_runtime', ['--config', config, '--input', visible,
          '--pool', pool, '--pool-sha256', digest(pool), '--output', report,
          '--device', device], 'runtime.log', module=True)
    result = json.loads(report.read_text(encoding='utf-8'))
    if (digest(config) != config_sha or digest(token_path) != token_ref['sha256']
            or result.get('schema') != 'rhyme17-visible-system-generation-1'
            or any(result.get(key) != expected for key, expected in expected_runtime.items())):
        raise ValueError('Runtime result differs from frozen request config/checkpoint/tokenizer/vocabulary')
    if result.get('complete') is not True or len(result.get('samples', [])) != 1:
        raise ValueError('Runtime did not produce a complete single-case report')
    sample = result['samples'][0]
    if sample['case'] != payload['cases'][0]:
        raise ValueError('Runtime report does not match the submitted input')
    selected = sample.get('selected')
    if selected is None:
        print('No continuation selected: ' + sample['status'])
        status = 1
    else:
        continuation = selected['text']
        if sample['status'] != 'selected' or not isinstance(continuation, str) or not continuation.strip():
            raise ValueError('Invalid selected continuation in runtime report')
        print(payload['cases'][0]['first_line'])
        print(continuation, end='' if continuation.endswith('\n') else '\n')
        status = 0
    print('Artifacts: ' + folder.relative_to(ROOT).as_posix())
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', required=True, help='First line or multiline poetry context')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    sys.exit(run(args.text, args.config, args.device))
