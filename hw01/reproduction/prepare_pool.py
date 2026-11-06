"""Host adapter for unchanged, hash-bound phonetic pool builders."""
import argparse
import hashlib
import importlib
from pathlib import Path
import sys

BUILDERS = ('scripts/prepare_rhyme17_runtime_pool.py', 'scripts/prepare_rhyme22_runtime_pool.py')


def prepare(builder_path, input_path, output):
    if builder_path not in BUILDERS:
        raise ValueError('Unsupported pool builder')
    builder = importlib.import_module(builder_path[:-3].replace('/', '.'))
    if sys.platform == 'win32':
        return builder.prepare(input_path, output)
    from reproduction.phonetics import load_native_tool
    from scripts.rhyme_endings16 import compare_endings

    def phonetics():
        tool, torch, manifest = load_native_tool()
        if builder_path == BUILDERS[1]:
            compare = builder.make_comparator(tool, torch.inference_mode)
        else:
            def compare(left, right):
                with torch.inference_mode():
                    return compare_endings(tool, left, right)
        host = dict(manifest['native_host'],
                    pool_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return compare, dict(rpst_revision=manifest['rpst_revision'],
                             rpst_version=manifest['rpst_version'], device='cpu',
                             native_host=host)

    if builder_path == BUILDERS[0]:
        return builder.prepare(input_path, output, phonetics_factory=phonetics)
    previous = builder.load_phonetics
    try:
        builder.load_phonetics = phonetics
        return builder.prepare(input_path, output)
    finally:
        builder.load_phonetics = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builder', choices=BUILDERS, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.builder, args.input, args.output))


if __name__ == '__main__':
    main()
