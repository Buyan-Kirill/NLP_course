"""Run the verified inference code with a portable, audited memory policy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from scripts import generate_rhyme17_system as runtime
from scripts import shared_gpu11 as shared


def admit_local(torch, minimum_free_mib, audit):
    """Require enough free memory; the training machine's total VRAM is irrelevant."""
    measurements = None
    identity = dict(policy='local_inference_free_memory',
                    policy_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        index = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        if not 0 <= free <= total or total <= 0:
            raise RuntimeError('Invalid CUDA memory telemetry')
        available = free / 1024**2
        measurements = dict(gpu_index=index, gpu_name=torch.cuda.get_device_name(index),
                            visible_devices=torch.cuda.device_count(),
                            free_mib=available, total_mib=total / 1024**2)
        if available < minimum_free_mib:
            raise RuntimeError(f'Insufficient free GPU memory: require {minimum_free_mib} MiB')
    except Exception as error:
        audit.record('admission_rejected', minimum_free_mib=minimum_free_mib,
                     measurements=measurements, error=repr(error), **identity)
        raise
    audit.record('admission_granted', minimum_free_mib=minimum_free_mib,
                 measurements=measurements, **identity)


def run(config, input_path, pool, output, device, *, pool_sha256):
    """Preserve frozen numerical sources and record the local host adapter separately."""
    output = runtime.local(output)
    receipt = output.with_suffix('.local_runtime.json')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf8') as stream:
        json.dump(dict(local=True, device=device, command=list(sys.argv),
                       policy_sha256=runtime.sha256(Path(__file__)),
                       runtime_sha256=runtime.sha256(Path(runtime.__file__))), stream, indent=2)
    previous = shared.admit
    try:
        shared.admit = admit_local
        return runtime.run(config, input_path, pool, output, device, pool_sha256=pool_sha256)
    finally:
        shared.admit = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'input', 'pool', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--pool-sha256', required=True)
    args = parser.parse_args()
    print(run(args.config, args.input, args.pool, args.output, args.device,
              pool_sha256=args.pool_sha256))


if __name__ == '__main__':
    main()
