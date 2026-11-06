"""Bind displayed continuations to stored model outputs and existing reader scores."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'audit/continuation19/final_reading/'
SOURCES = {
    BASE+'mapping.json': 'Привязка оценок к исходным генерациям',
    BASE+'cards.json': 'Инструкции слепого чтения',
    BASE+'judged_primary.json': 'Оценки первого агента по каждому критерию',
    BASE+'judged_secondary.json': 'Оценки второго агента по каждому критерию',
    'audit/continuation19/inference/final_pair/first.json': 'Исходные генерации заключительной проверки',
    'docs/generation_quality_rubric_v2.md': 'Согласованная десятибалльная шкала',
}
AXES = ['relevance', 'progress', 'logic', 'naturalness', 'style']


def read(name):
    return json.loads((ROOT/name).read_text('utf8'))


def fingerprint(name):
    return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


def final_examples():
    release = read('artifacts/releases/poetry17/manifest.json')
    mapping = read(BASE+'mapping.json')
    source_name = 'audit/continuation19/inference/final_pair/first.json'
    assert fingerprint(source_name) == mapping['sources_sha256'][source_name]
    original = read(source_name)
    assert original['complete'] and not original['training']
    assert len(original['models']) == 1 and original['models'][0]['label'] == 'planned400k'
    assert original['models'][0]['sha256'] == release['source']['sha256']
    cards = {r['id']: r for r in read(BASE+'cards.json')['cards']}
    readers = {key: {r['id']: r for r in read(BASE+f'judged_{key}.json')['rows']}
               for key in ['primary','secondary']}
    examples = {}
    for row in mapping['rows']:
        assert row['source'] == source_name and row['model'] == 'planned400k'
        sample = original['samples'][row['sample_index']]
        value = sample
        for field in row['quote_path']:
            value = value[field] if value is not None else None
        assert value == row['quote'] == cards[row['id']]['text']
        assert sample['case']['prefix'] == row['prefix']
        ratings = []
        for reader, judged in readers.items():
            rating = judged[row['id']]
            assert rating['quote'] == value
            assert all(type(rating['scores'][axis]) is int and 0 <= rating['scores'][axis] <= 2
                       for axis in AXES)
            assert rating['total'] == sum(rating['scores'][axis] for axis in AXES)
            ratings.append(dict(reader=reader, total=rating['total'], scores=rating['scores']))
        example = examples.setdefault(row['case_id'], dict(id=row['case_id'], prefix=row['prefix']))
        example[row['mode']] = dict(text=value, ratings=ratings, source_index=row['sample_index'])
    result = list(examples.values())
    assert len(result) == 6 and all('free' in r and 'constrained' in r for r in result)
    for row in result:
        row.update(source=source_name, checkpoint_sha256=release['source']['sha256'])
    return result


def verify_examples(data):
    """Compare every displayed character and binding; references are never outputs."""
    checked = []
    release = read('artifacts/releases/poetry17/manifest.json')
    hero = data['hero']
    raw = read(hero['source'])
    assert raw['generation'] and not raw['training']
    assert raw['checkpoint']['sha256'] == release['continuation']['sha256']
    assert raw['samples'][2]['selected']['text'].strip() == hero['output']
    checked.append(dict(kind='hero', source=hero['source'], sample_index=2,
                        checkpoint_sha256=raw['checkpoint']['sha256']))
    assert release['source_lineage']['origin'] == 'random_initialization'
    expected = [p['checkpoint_sha256'] for p in release['source_lineage']['parents'][1:]]
    expected.append(release['source']['sha256'])
    assert len(expected) == len(data['stages'])
    for index, stage in enumerate(data['stages']):
        for example, source_index in zip(stage['examples'], [0,4,8]):
            raw = read(example['source'])
            sample = raw['samples'][source_index]
            assert raw['complete'] and not raw['training']
            assert raw['model']['n_layers'] == 20 and raw['model']['d_model'] == 512
            assert example['text'] == sample['text'] and example['prompt'] == sample['case']['prompt']
            assert example['seed'] == sample['seed']
            assert raw['checkpoint_sha256'] == expected[index]
            checked.append(dict(kind='stage', stage=stage['label'], source=example['source'],
                sample_index=source_index, checkpoint_sha256=raw['checkpoint_sha256']))
    for row in data['open_examples']:
        raw = read(row['source'])
        assert raw['models'][0]['sha256'] == release['source']['sha256']
        sample = raw['samples'][row['index']]
        assert row['prefix'] == sample['case']['prefix']
        assert row['free'] == sample['free']['text']
        assert row['text'] == (sample['constrained'] or {}).get('text')
    for row in data['examples']:
        for mode in ['free', 'constrained']:
            checked.append(dict(kind=mode, source=row['source'],
                sample_index=row[mode]['source_index'], checkpoint_sha256=row['checkpoint_sha256']))
    return dict(complete=True, displayed_views=len(checked), views=checked,
        checks=['exact raw output fields, not literary reference fields',
                'checkpoint SHA matches selected model lineage', 'judge quotes and score sums'],
        new_generations=False, historical_scores_modified=False)
