"""Fit a separate byte-level BPE experiment using training texts only.

This does not replace the selected tokenizer or retokenize existing weights.
"""
import argparse
import json
from pathlib import Path

from reproduction.download import ROOT, sha256
from reproduction.prepare import JsonlRows


def fit(output, paths, vocab_size=8192):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from poetrygpt.serialization import SPECIAL_TOKENS

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    paths = [Path(path) for path in paths]
    if any(path.name not in ('train.jsonl', 'expanded_train.jsonl') for path in paths):
        raise ValueError('Use explicit train.jsonl/expanded_train.jsonl files; validation/test are forbidden')
    inputs = {str(path): sha256(path) for path in paths}

    def texts():
        for path in paths:
            for row in JsonlRows(path):
                yield row['text']

    model = Tokenizer(models.BPE())
    model.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    model.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
        special_tokens=SPECIAL_TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    model.train_from_iterator(texts(), trainer=trainer)
    output.mkdir(parents=True)
    path = output / 'tokenizer.json'
    model.save(str(path))
    for text in ['В саду поёт соловей.\nИ меркнет свет.\n', 'ѣ Ё ъ — …', 'слово\n\nстрока']:
        if model.decode(model.encode(text).ids, skip_special_tokens=False) != text:
            raise ValueError('Byte-level tokenizer roundtrip failed')
    report = dict(inputs_sha256=inputs, tokenizer_sha256=sha256(path),
                  vocab_size=model.get_vocab_size(), special_tokens=SPECIAL_TOKENS)
    (output / 'config.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n','utf8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vocab-size', type=int, default=8192)
    args = parser.parse_args()
    print(json.dumps(fit(args.output, args.input, args.vocab_size), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
