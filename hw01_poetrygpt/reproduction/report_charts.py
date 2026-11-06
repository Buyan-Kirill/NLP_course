"""Training curves from saved logs; no model execution or inferred observations."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    'reproduction/evidence/d_initial_metrics.jsonl': 'Полный журнал D: 630,19 млн токенов',
    'reproduction/evidence/lr50_metrics.jsonl': 'Журнал завершения претрейна LR50',
    'reproduction/evidence/poetry_first_metrics.jsonl': 'Журнал первого прохода поэзии',
    'reproduction/evidence/native_sft_metrics.jsonl': 'Журнал SFT продолжения строки',
    'audit/poetry16/batch4k/2423668_validation.json': 'Validation в середине второго прохода поэзии',
    'audit/pretrain11/lr25_interim_check.json': 'Контрольная панель претрейна: D630, LR10 и LR25',
}


def read_log(path):
    """Resolve rolled-back steps before aggregation; retain the latest valid write."""
    train, validation = {}, {}
    for line in path.read_text('utf8').splitlines():
        row = json.loads(line)
        rollback = row.get('rollback_after_step')
        if row.get('event') in ('start', 'resume') and rollback is not None:
            train = {step: value for step, value in train.items() if step <= rollback}
            validation = {step: value for step, value in validation.items() if step <= rollback}
        if row.get('event') == 'train':
            train[row['step']] = row
        elif row.get('event') == 'validation':
            validation[row['step']] = row
    return [train[k] for k in sorted(train)], [validation[k] for k in sorted(validation)]


def token_count(row):
    for key in ('local_targets', 'tokens_seen', 'total_supervised_tokens'):
        if key in row:
            return row[key]
    raise ValueError('Missing cumulative token count')


def aggregate_train(rows, maximum_points=120):
    """Non-overlapping step blocks, weighted by the number of supervised tokens."""
    width = max(1, math.ceil(len(rows) / maximum_points))
    result = []
    for index in range(0, len(rows), width):
        chunk = rows[index:index+width]
        weights = [r.get('supervised_targets', r.get('targets')) for r in chunk]
        if not all(isinstance(w, int) and w > 0 for w in weights):
            raise ValueError('Invalid loss denominator')
        result.append(dict(x=token_count(chunk[-1]), step=chunk[-1]['step'],
            nll=sum(r['nll'] * w for r, w in zip(chunk, weights)) / sum(weights),
            lr=chunk[-1]['lr'], steps=len(chunk), tokens=sum(weights)))
    return result, width


def series(label, rows, metric, *, color, dashed=False, connect=True):
    points = [dict(x=r['x'], y=r[metric], step=r.get('step')) for r in rows]
    assert points and all(math.isfinite(p['y']) for p in points)
    assert all(a['x'] < b['x'] for a, b in zip(points, points[1:]))
    return dict(label=label, points=points, color=color, dashed=dashed, connect=connect)


def logged_history(key, label, source, description, *, native_validation=None):
    train, validation = read_log(ROOT/source)
    averaged, width = aggregate_train(train)
    by_step = {r['step']: token_count(r) for r in train}
    val = []
    for row in validation:
        if 'domains' in row:
            metric = row['domains']['poetry']
        else:
            metric = row
        val.append(dict(x=by_step[row['step']], step=row['step'], nll=metric['nll'],
                        perplexity=math.exp(metric['nll'])))
    if native_validation:
        val = [dict(x=0, nll=native_validation[0]['nll'],
                    perplexity=native_validation[0]['perplexity']),
               dict(x=token_count(train[-1]), nll=native_validation[1]['nll'],
                    perplexity=native_validation[1]['perplexity'])]
    curves = {
        'nll': [series('Train NLL', averaged, 'nll', color='#256855')],
        'lr': [series('Learning rate', averaged, 'lr', color='#6d599b')],
    }
    if val:
        curves['nll'].append(series('Validation NLL', val, 'nll', color='#ad5a25',
                                   dashed=True, connect=not bool(native_validation)))
        curves['perplexity'] = [series('Validation PPL', val, 'perplexity', color='#ad5a25',
                                      connect=not bool(native_validation))]
    return dict(id=key, label=label, description=description,
        scale=1e6 if token_count(train[-1]) > 1e6 else 1000,
        unit='млн токенов' if token_count(train[-1]) > 1e6 else 'тыс. токенов',
        note=f'Train: среднее по блокам до {width} шагов, взвешенное по токенам. '
             + ('Validation: только замеры до и после обучения.' if native_validation else
                'Validation: сохранённые замеры без сглаживания.'),
        source=source, curves=curves)


def histories(poetry_rows, native):
    initial = logged_history('initial', 'Претрейн D · 0–630,19 млн токенов',
        'reproduction/evidence/d_initial_metrics.jsonl',
        'Три прохода корпуса: 630 193 557 обучающих токенов, включая EOS. '
        'Validation: 261 391 токен, 475 записей, 85 семейств. Полный журнал выбранной ветви D.')
    assert initial['curves']['nll'][0]['points'][-1]['x'] == 630193557
    assert initial['curves']['perplexity'][0]['points'][-1]['x'] == 630193557
    finish = logged_history('lr50', 'Завершение претрейна · LR50',
        'reproduction/evidence/lr50_metrics.jsonl',
        'Ещё 50 млн токенов от D630, LR ×0,5. Validation: 261 391 токен; '
        'ось X отсчитывается от начала этого продолжения.')
    panel_source = 'audit/pretrain11/lr25_interim_check.json'
    interim = json.loads((ROOT/panel_source).read_text('utf8'))
    assert interim['matched_panel_settings_prefixes_seeds']
    final = json.loads((ROOT/'audit/pretrain11/final_automatic_review.json').read_text('utf8'))
    baseline = final['main']['models']['D630']
    selected = final['main']['models']['LR50']
    points = []
    for label in ['D630','LR10','LR25']:
        row = interim['rows'][label]
        points.append(dict(x=row['absolute_targets'] - interim['rows']['D630']['absolute_targets'],
            nll=row['validation_nll'], perplexity=math.exp(row['validation_nll']),
            accuracy=row['validation_token_accuracy']))
    assert points[0]['nll'] == baseline['teacher_forced']['validation']['nll']
    assert points[0]['accuracy'] == baseline['teacher_forced']['validation']['token_accuracy']
    val = selected['teacher_forced']['validation']
    points.append(dict(x=50000000, nll=val['nll'], perplexity=val['perplexity'],
        accuracy=val['token_accuracy']))
    quality = dict(id='quality', label='LR50 · контроль NLL / PPL / Top-1', scale=1e6, unit='млн токенов',
        description='Дополнительные замеры завершения претрейна LR50 на 0 / 10 / 25 / 50 млн токенов: '
                    '256 validation-окон, 16 384 токена.',
        note='Top-1 измерен на эталонном контексте. '
             'Линии соединяют четыре измеренных checkpoint. Эта панель отличается от validation тренировочного журнала.',
        source=panel_source, curves={
            key: [series(label, points, key, color=color)] for key, label, color in [
                ('nll','Validation NLL','#ad5a25'), ('perplexity','Validation PPL','#ad5a25'),
                ('accuracy','Top-1','#256855')]})
    first = logged_history('poetry_first', 'Поэзия · журнал первого прохода',
        'reproduction/evidence/poetry_first_metrics.jsonl',
        'Первый из двух проходов выбранной поэтической адаптации. '
        'Validation: 85 928 токенов, 324 записи, 323 семейства.')
    train, validation = read_log(ROOT/'reproduction/evidence/poetry_first_metrics.jsonl')
    by_step = {r['step']: token_count(r) for r in train}
    midpoint = json.loads((ROOT/'audit/poetry16/batch4k/2423668_validation.json').read_text('utf8'))
    middle = midpoint['domains']['poetry']['metrics']['overall']
    points = [dict(x=0, nll=poetry_rows[0]['nll'], perplexity=poetry_rows[0]['perplexity'])]
    points.extend(dict(x=by_step[r['step']], nll=r['nll'], perplexity=r['perplexity']) for r in validation)
    points.extend([
        dict(x=2423668, nll=middle['nll'], perplexity=middle['perplexity']),
        dict(x=3231558, nll=poetry_rows[3]['nll'], perplexity=poetry_rows[3]['perplexity']),
        dict(x=3631558, nll=poetry_rows[4]['nll'], perplexity=poetry_rows[4]['perplexity']),
    ])
    poetry = dict(id='poetry', label='Поэзия · два прохода + обучение с окончанием', scale=1e6, unit='млн токенов',
        description='Общая поэтическая validation: 85 928 токенов. Два прохода, затем '
                    '400 тыс. токенов со смесью обычной поэзии и подсказок окончания.',
        note='Точки — измеренные checkpoint. Линии соединяют соседние замеры; '
             'промежуточные значения не измерялись.',
        source='audit/poetry16/batch4k/2423668_validation.json',
        boundaries=[dict(x=1615779, label='Второй проход'), dict(x=3231558, label='Окончание')],
        curves={key: [series('Validation NLL' if key=='nll' else 'Validation PPL', points,
                             key, color='#ad5a25')] for key in ['nll','perplexity']})
    sft = logged_history('native', 'SFT продолжения · два прохода',
        'reproduction/evidence/native_sft_metrics.jsonl',
        'Контрольная ветвь от Planned400k; в выбранные веса не входит. '
        '869 фрагментов, два прохода, 21 680 токенов в loss. '
        'Validation: 282 токена продолжений, 23 записи, 21 семейство.', native_validation=native)
    selected = [initial, finish, poetry]
    details = [quality, first]
    for group, entries in [
        ('Выбранная модель · порядок обучения', selected),
        ('Подробные замеры этих этапов', details),
        ('Отдельная ветвь · вне выбранной модели', [sft]),
    ]:
        for entry in entries:
            entry['group'] = group
    return [*selected, *details, sft]
