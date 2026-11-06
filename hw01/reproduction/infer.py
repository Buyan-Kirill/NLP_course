"""Run both rhyme preparation and the language model locally; no SSH required."""
import argparse
from pathlib import Path


def main():
    from scripts.continue_rhyme17 import run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', required=True, help='First verse line (an optional acute sets final-word stress)')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--config', type=Path, default=Path('configs/inference.json'))
    args = parser.parse_args()
    from scripts.try_poetry import resolve_device
    return run(args.text, args.config, resolve_device(args.device))


if __name__ == '__main__':
    raise SystemExit(main())
