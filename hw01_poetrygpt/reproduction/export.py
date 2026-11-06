"""Export a reproduced final checkpoint for the same notebook interface."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from reproduction.download import ROOT, sha256


def local(path):
    path = (ROOT / path).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError('Use a path inside this project')
    return path


def export(checkpoint, output):
    import torch
    from poetrygpt.runtime import load_checkpoint

    source, output = local(checkpoint), local(output)
    if output.exists():
        raise FileExistsError('Preserve existing export: ' + str(output))
    source_sha = sha256(source)
    state = load_checkpoint(source)
    cfg = json.loads((ROOT / 'configs/inference.json').read_text('utf8'))
    if (state.get('lineage', {}).get('origin') != 'random_initialization'
            or state.get('tokens_seen') != 400000
            or state['tokenizer_sha256'] != cfg['tokenizer']['sha256']):
        raise ValueError('Expected the completed ending_plan stage and selected tokenizer')
    selected_model = json.loads((ROOT / 'configs/pretrain.json').read_text('utf8'))['model']
    if state['model_config'] != selected_model:
        raise ValueError('Checkpoint architecture differs from the selected training route')
    exported = {key: state[key] for key in
                ['format', 'model_config', 'model', 'tokenizer_sha256', 'config',
                 'step', 'tokens_seen', 'lineage']}
    exported.update(inference_only=True, source_checkpoint_sha256=source_sha)
    output.mkdir(parents=True)
    weights = output / 'continuation.pt'
    torch.save(exported, weights)
    restored = load_checkpoint(weights)
    if any(not torch.equal(value, restored['model'][name]) for name, value in state['model'].items()):
        raise ValueError('Exported model tensors differ')
    if sha256(source) != source_sha:
        raise ValueError('Training checkpoint changed during export')
    config = deepcopy(cfg)
    config['checkpoint'] = dict(path=weights.relative_to(ROOT).as_posix(), sha256=sha256(weights))
    config['label'] = 'locally-reproduced-ending-plan'
    config_path = output / 'system_config.json'
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', 'utf8')
    return config_path.relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/reproduction/ending_plan/last.pt'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/local'))
    args = parser.parse_args()
    print(export(args.checkpoint, args.output))


if __name__ == '__main__':
    main()
