"""Portable resource checks around the original numerical training entrypoints."""
import argparse
import importlib
from pathlib import Path
import sys

TARGETS = {'scripts/train_pretrain08.py', 'scripts/train_targeted10.py',
           'scripts/train_poetry08.py', 'scripts/train_poetry16.py',
           'scripts/train_poetry_units16.py'}


def execute(target, arguments):
    from reproduction.download import ROOT, sha256
    from reproduction.local_runtime import admit_local
    from scripts import shared_gpu11 as shared
    import torch

    # Import aliases before replacing their admission checks. Optimizers,
    # loss, samplers and checkpoint validation are unchanged.
    modules = [importlib.import_module(name[:-3].replace('/', '.')) for name in sorted(TARGETS)]
    audit = shared.Audit(adapter_sha256=sha256(Path(__file__)), phase='reproduction_training')
    def admit_training(torch_module, minimum, record):
        shared.AUDIT = ROOT / 'runs/reproduction/resource_admission.jsonl'
        record.identity['training_adapter_sha256'] = sha256(Path(__file__))
        return admit_local(torch_module, minimum, record)

    shared.admit = admit_training

    def check():
        admit_training(torch, 10240, audit)

    for module in modules:
        if hasattr(module, 'check_gpu_idle'):
            module.check_gpu_idle = check
    module = importlib.import_module(target[:-3].replace('/', '.'))
    previous = sys.argv
    try:
        sys.argv = [str(ROOT / target), *arguments]
        module.main()
    finally:
        sys.argv = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', choices=sorted(TARGETS))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    execute(args.target, args.arguments)


if __name__ == '__main__':
    main()
