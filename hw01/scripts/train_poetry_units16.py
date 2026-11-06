"""Matched fresh-stage experiments: source-aware windows and contextual AA loss.

The underlying CE, model, Muon/AdamW implementation and validation stay unchanged.
Every branch starts fresh optimizers from its declared poetry parent. Exact resume
binds the wrapper, sampler, auxiliary masks, source windows and original corpus.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tokenizers import Tokenizer
from poetrygpt.data import PreparedDataset, read_json, sha256
from scripts import train_poetry08 as trainer
from scripts import shared_gpu11 as shared
from scripts.poetry16_streams import TargetRecordStream, UnitWindowStream, PoetryAuxStream


def checked(reference):
    if set(reference) != {'path', 'sha256'}:
        raise ValueError('Expected path/SHA reference')
    path = (ROOT / reference['path']).resolve()
    if not path.is_relative_to(ROOT) or sha256(path) != reference['sha256']:
        raise ValueError('Changed or external experiment input: ' + str(path))
    return path


def configuration(raw):
    if set(raw) != {'training', 'parent_sha256', 'windows', 'auxiliary'}:
        raise ValueError('Unexpected unit experiment fields')
    cfg = trainer.config(raw['training'])
    if cfg['mode'] != 'pure' or cfg['optimizer'] != 'muon':
        raise ValueError('Matched pure-poetry Muon stage required')
    checked(dict(path=cfg['parent'], sha256=raw['parent_sha256']))
    if raw['windows'] is not None:
        checked(raw['windows'])
    aux = raw['auxiliary']
    if aux is not None:
        if set(aux) != {'manifest', 'numerator', 'denominator'}:
            raise ValueError('Unexpected auxiliary fields')
        checked(aux['manifest'])
        n, d = aux['numerator'], aux['denominator']
        if type(n) is not int or type(d) is not int or not 0 < n < d:
            raise ValueError('Invalid auxiliary target fraction')
    return cfg


@contextmanager
def experiment_context(raw):
    old_validate, old_identity, old_stream = trainer.validate_checkpoint, trainer.input_identity, trainer.MixtureStream
    cfg = configuration(raw)
    windows = None if raw['windows'] is None else np.load(checked(raw['windows']), allow_pickle=False)
    auxiliary = None
    if raw['auxiliary'] is not None:
        folder = checked(raw['auxiliary']['manifest']).parent
        auxiliary = PreparedDataset(folder, 'sft', 'train', cfg['model']['context_length'])
        token_hash = sha256(cfg['tokenizer'])
        declared = auxiliary.manifest['inputs_sha256']
        if token_hash not in [v for k, v in declared.items() if k.replace('\\', '/').endswith('tokenizer.json')]:
            raise ValueError('Auxiliary tokenizer mismatch')
        tokenizer = Tokenizer.from_file(cfg['tokenizer'])
        if any(tokenizer.token_to_id(name) != value for name, value in auxiliary.manifest['special_tokens'].items()):
            raise ValueError('Auxiliary special token IDs mismatch')

    def validate(state, config, tokenizer_sha, *, initial=False):
        old_validate(state, config, tokenizer_sha, initial=False)
        if initial and (state['config'].get('optimizer') != 'muon' or
                        set(state.get('optimizers', {})) != {'muon', 'adamw'}):
            raise ValueError('Expected our full poetry checkpoint')

    def identity(*args, **kwargs):
        result = old_identity(*args, **kwargs)
        result['poetry16_units'] = dict(recipe=deepcopy(raw), wrapper_sha256=sha256(__file__),
            streams_sha256=sha256(ROOT / 'scripts/poetry16_streams.py'),
            auxiliary_files_sha256=None if auxiliary is None else auxiliary.hashes,
            optimizer_start='fresh for every compared branch; exact state restored on resume',
            objective='ordinary CE; optional contextual second-line targets, no invented EOS')
        return result

    def stream(datasets, seed, prose_numerator, prose_denominator=10, state=None):
        if set(datasets) != {'poetry'} or prose_numerator != 0:
            raise ValueError('Unexpected main training domains')
        saved = {} if state is None else state['domains']
        if windows is None:
            main = trainer.EpochStream(datasets['poetry'], seed, saved.get('poetry'))
        else:
            main = UnitWindowStream(datasets['poetry'], seed, saved.get('poetry'), windows=windows)
        aux = None if auxiliary is None else TargetRecordStream(auxiliary, seed+1, saved.get('auxiliary'))
        fraction = raw['auxiliary'] or dict(numerator=0, denominator=5)
        return PoetryAuxStream(main, aux, numerator=fraction['numerator'], denominator=fraction['denominator'], state=state)

    trainer.validate_checkpoint, trainer.input_identity, trainer.MixtureStream = validate, identity, stream
    try:
        yield cfg
    finally:
        trainer.validate_checkpoint, trainer.input_identity, trainer.MixtureStream = old_validate, old_identity, old_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--until-tokens', type=int, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    raw = read_json(args.config)
    if args.device == 'cuda':
        shared.AUDIT = ROOT / 'audit/poetry16/gpu_admission.jsonl'
        trainer.check_gpu_idle = lambda: shared.admit(trainer.torch, 10240,
            shared.Audit(coordinator_sha256=sha256(__file__), phase='poetry16_units'))
    with experiment_context(raw) as cfg:
        trainer.train(cfg, args.run_dir, args.until_tokens, args.device, args.resume, args.stop_after)


if __name__ == '__main__':
    main()
