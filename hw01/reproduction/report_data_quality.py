"""Bind corpus statistics and historical BPE comparisons to their frozen inputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'reproduction/evidence/corpus_statistics.json': 'Частоты слов в окончательном train',
    'reproduction/evidence/tokenizers_v2.json': 'Шесть опытов BPE: 4k / 8k / 16k и два состава train',
    'reproduction/evidence/tokenizers_v3.json': 'Повторная проверка BPE на раздельных доменах',
}


def read(name):
    return json.loads((ROOT / name).read_text('utf-8-sig'))


def quality_data():
    corpus = read('reproduction/evidence/corpus_statistics.json')
    catalog = read('reproduction/corpus/manifest.json')
    for domain, relative in [('prose', 'data/prose_v7/expanded_train.jsonl'),
                             ('poetry', 'data/poetry_v7/train.jsonl')]:
        expected = catalog['domains'][domain]['outputs'][relative]['sha256']
        assert corpus['domains'][domain]['input_sha256'] == expected
    earlier = read('reproduction/evidence/tokenizers_v2.json')
    later = read('reproduction/evidence/tokenizers_v3.json')
    tokenizer = ROOT / 'artifacts/tokenizers_v3/selected/tokenizer.json'
    assert hashlib.sha256(tokenizer.read_bytes()).hexdigest() == later['selected_sha256']
    assert earlier['selected'] == 'bpe_8192_mixed' and later['selected'] == 'baseline_v2_8192'
    rows = []
    for trial in earlier['experiments']:
        train, validation = trial['pretrain_train'], trial['pretrain_validation']
        rows.append(dict(name=trial['name'], size=trial['vocab_actual'], mixture=trial['mixture'],
            tokens_per_word=validation['tokens_per_word'],
            rare_types=train['types_used_1_to_5_times'], unused_types=train['unused_vocab_types'],
            selected=trial['name'] == earlier['selected']))
    domains = []
    for trial in later['experiments']:
        domains.append(dict(name=trial['name'], size=trial['vocab_actual'],
            **{domain: metrics['tokens_per_word'] for domain, metrics in trial['validation'].items()}))
    return dict(corpus=corpus, tokenizers=rows, tokenizers_by_domain=domains,
                decision=later['decision'], selected_sha256=later['selected_sha256'])
