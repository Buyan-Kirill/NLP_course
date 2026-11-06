"""Audited resource admission for three frozen series11 child entrypoints.

Only the target module's local admission callable is substituted. Source files,
native locks, model/data validation and numerical execution remain unchanged.
The separate audit is part of execution provenance, beyond frozen fingerprints.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUDIT = ROOT / 'audit/pretrain11/shared_gpu_admission.jsonl'
TARGETS = {
    'scripts/evaluate_learning_series04.py': 3072,
    'scripts/evaluate_causal_series.py': 3072,
    'scripts/train_rebalance11.py': 8192,
}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Audit:
    """Append-only records; failure to persist a record prevents execution."""
    def __init__(self, **identity):
        self.identity = dict(attempt=uuid.uuid4().hex, pid=os.getpid(), **identity)

    def record(self, event, **details):
        row = dict(self.identity, utc=datetime.now(timezone.utc).isoformat(),
                   event=event, **details)
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def gpu_measurements(torch):
    """Require one unambiguous physical/visible GPU; fail closed on bad telemetry."""
    command = ['nvidia-smi',
        '--query-gpu=index,uuid,memory.total,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError('Shared admission requires exactly one physical GPU')
    fields = [value.strip() for value in rows[0].split(',')]
    if len(fields) != 6 or not fields[1].startswith('GPU-'):
        raise RuntimeError('Invalid nvidia-smi GPU record')
    try:
        index, total, free, used, utilization = map(int, [fields[0], *fields[2:]])
    except ValueError as error:
        raise RuntimeError('Invalid numeric nvidia-smi telemetry') from error
    if (index != 0 or total <= 0 or not 0 <= free <= total
            or not 0 <= used <= total or not 0 <= utilization <= 100):
        raise RuntimeError('Out-of-range nvidia-smi telemetry')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None and visible not in ('0', fields[1]):
        raise RuntimeError('Ambiguous CUDA_VISIBLE_DEVICES mapping')
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError('Shared admission requires exactly one visible CUDA GPU')
    cuda_free, cuda_total = torch.cuda.mem_get_info(0)
    cuda_free, cuda_total = cuda_free / 1024**2, cuda_total / 1024**2
    if not (0 <= cuda_free <= cuda_total and abs(cuda_total - total) <= 64):
        raise RuntimeError('CUDA and nvidia-smi memory telemetry disagree')
    return dict(nvidia_command=command, gpu_uuid=fields[1], gpu_index=index,
                total_mib=total, free_mib=free, used_mib=used,
                utilization_percent=utilization, cuda_free_mib=cuda_free,
                cuda_total_mib=cuda_total)


def admit(torch, minimum_free_mib, audit):
    measurements = None
    try:
        measurements = gpu_measurements(torch)
        # NVIDIA exposes slightly less than the nominal 16 GiB on this machine.
        if not 15.5 * 1024 <= measurements['total_mib'] <= 16.5 * 1024:
            raise RuntimeError('Shared series11 admission expects a nominal 16 GiB GPU')
        if min(measurements['free_mib'], measurements['cuda_free_mib']) < minimum_free_mib:
            raise RuntimeError(f'Insufficient free GPU memory: require {minimum_free_mib} MiB')
    except Exception as error:
        audit.record('admission_rejected', minimum_free_mib=minimum_free_mib,
                     measurements=measurements, error=repr(error))
        raise
    audit.record('admission_granted', minimum_free_mib=minimum_free_mib,
                 measurements=measurements, utilization_policy='record only')


def execute(target, arguments, expected_target_sha256):
    if target not in TARGETS:
        raise ValueError('Unsupported shared GPU target: ' + target)
    target_path = ROOT / target
    digest = file_sha(target_path)
    audit = Audit(wrapper_sha256=file_sha(__file__), target=target, target_sha256=digest,
                  expected_target_sha256=expected_target_sha256,
                  command=[sys.executable, target, *arguments], wrapper_command=list(sys.argv),
                  admission_helpers_sha256={name: file_sha(ROOT / name) for name in
                      ('scripts/train_causal_series.py', 'scripts/run_task_pilot02.py')})
    audit.record('execution_started')
    attribute = original = module = None
    previous_argv = sys.argv
    try:
        if digest != expected_target_sha256:
            raise ValueError('Shared target source SHA256 mismatch')
        module = importlib.import_module(target[:-3].replace('/', '.'))
        if Path(module.__file__).resolve() != target_path.resolve() or file_sha(target_path) != digest:
            raise ValueError('Imported target differs from the audited source')
        if target == 'scripts/evaluate_causal_series.py':
            attribute = 'require_resources'
            original = getattr(module, attribute)

            def resource_check(training):
                if training is not False:
                    raise ValueError('Quality wrapper only supports inference')
                admit(module.torch, TARGETS[target], audit)
                return original(training)
        else:
            attribute = 'check_gpu_idle'
            original = getattr(module, attribute)

            def resource_check():
                admit(module.torch, TARGETS[target], audit)

        setattr(module, attribute, resource_check)
        sys.argv = [str(target_path), *arguments]
        module.main()
    except BaseException as error:
        audit.record('execution_finished', success=isinstance(error, SystemExit) and error.code in (0, None),
                     error=repr(error))
        raise
    else:
        audit.record('execution_finished', success=True)
    finally:
        sys.argv = previous_argv
        if attribute is not None and original is not None:
            setattr(module, attribute, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-target-sha256', required=True)
    parser.add_argument('target', choices=TARGETS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    execute(args.target, args.arguments, args.expected_target_sha256)


if __name__ == '__main__':
    main()
