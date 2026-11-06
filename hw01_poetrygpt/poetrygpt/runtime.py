"""Portable single-device runtime and trusted project checkpoint helpers."""
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import platform
import random

import numpy as np
import torch

from .data import sha256


@contextmanager
def run_lock(directory, kind):
    """Hold an OS lock; process death releases it without stale-PID cleanup."""
    directory = Path(directory).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    # A sibling avoids making a new run directory appear nonempty. Keep this
    # tiny file permanently: unlinking a lock file can race another opener.
    lock_path = directory.parent / f'.{directory.name}.{kind}.lock'
    with lock_path.open('a+b') as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b'\0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            lock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            lock()
        except OSError as error:
            raise RuntimeError(f'Run is already locked: {directory} ({kind})') from error
        try:
            yield
        finally:
            stream.seek(0)
            unlock()


def locked_run(argument, kind):
    """Protect the complete public entry point, including preflight writes."""
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            with run_lock(bound.arguments[argument], kind):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def repair_incomplete_jsonl_tail(path):
    """Remove only a final unterminated event left by a killed writer."""
    path = Path(path)
    if not path.exists():
        return 0
    with path.open('r+b') as stream:
        size = stream.seek(0, os.SEEK_END)
        if not size:
            return 0
        stream.seek(size - 1)
        if stream.read(1) == b'\n':
            return 0
        position = size
        end = 0
        while position:
            start = max(0, position - 8192)
            stream.seek(start)
            chunk = stream.read(position - start)
            newline = chunk.rfind(b'\n')
            if newline >= 0:
                end = start + newline + 1
                break
            position = start
        stream.truncate(end)
        return size - end


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def append_event(path, event):
    event = {'time_utc': datetime.now(timezone.utc).isoformat(), **event}
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    values = state['numpy']
    np.random.set_state((values[0], np.asarray(values[1], dtype=np.uint32), *values[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available():
            raise ValueError('CUDA checkpoint RNG requires CUDA for exact resume; use init-from for a new stage')
        torch.cuda.set_rng_state_all(state['cuda'])


def resolve_device(name, precision):
    name = ('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else name
    if name not in ('cpu', 'cuda') or (name == 'cuda' and not torch.cuda.is_available()):
        raise ValueError(f'Unavailable device: {name}')
    if precision == 'auto':
        precision = 'bf16' if name == 'cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    if precision not in ('fp32', 'bf16') or (precision == 'bf16' and (name != 'cuda' or not torch.cuda.is_bf16_supported())):
        raise ValueError('Use fp32, or bf16 on a CUDA device supporting it')
    return torch.device(name), precision


def autocast(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()


def environment(device, precision):
    return {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__,
            'platform': platform.platform(), 'device': str(device), 'precision': precision,
            'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'cpu_threads': torch.get_num_threads(), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}


def code_hashes():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    # Only tensors and primitive state types are used: no arbitrary pickle code.
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != 'poetrygpt-v1' or state.get('lineage', {}).get('origin') != 'random_initialization':
        raise ValueError('Expected a project checkpoint descending from random initialization')
    return state
