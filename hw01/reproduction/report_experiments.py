"""Compact experimental comparisons, bound to frozen source fields."""
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'reproduction/evidence/experiments.json'
SOURCES = {SOURCE: 'Сравнения корпуса, архитектур и методов: исходные поля и SHA протоколов'}


def experiment_data():
    data = json.loads((ROOT/SOURCE).read_text('utf8'))
    for row in data['architectures']:
        cfg = row['config']
        assert row['checkpoint']['tokens_seen'] == 420000000
        assert row['validation']['targets'] == 16384
        assert row['validation']['n_families'] == 73
        assert cfg['seed'] == 2026110308 and cfg['target_tokens'] == 16384
        assert cfg['model']['context_length'] == 512
        assert row['checks']['explicit_train_counter_and_validation_step_match']
        for reading in row['reading'].values():
            assert reading['early']['n'] == reading['full']['n'] == 23
    for row in data['corpus_matrix']:
        assert row['overall']['targets'] == 10200341
    for row in [*data['architectures'], *data['budgets'], *data['pilots']]:
        val = row['validation']
        assert math.isclose(math.exp(val['nll']), val['perplexity'], rel_tol=1e-6)
    return data


def table(headers, rows, caption):
    cells = lambda values, tag: ''.join(f'<{tag}>{html.escape(str(v))}</{tag}>' for v in values)
    return ('<div class="scroll"><table><caption>'+html.escape(caption)+'</caption><thead><tr>'
        +cells(headers,'th')+'</tr></thead><tbody>'
        +''.join('<tr>'+cells(row,'td')+'</tr>' for row in rows)+'</tbody></table></div>')


def score(row, view):
    readings = [entry[view] for entry in row['reading'].values()]
    # Missing ratings stay excluded from their original denominator.
    return f'{10*sum(r["raw_score"] for r in readings)/sum(r["max_denominator"] for r in readings):.2f}'


def render_tables(data):
    matrix = table(['Модель / корпус','NLL ↓','PPL ↓','NLL ядра ↓','NLL дополнения ↓'], [
        [row['label'].replace('17m','17,3M').replace('36m','36,3M').replace('_strict',' / узкий').replace('_expanded',' / расширенный'),
         f'{row["overall"]["nll"]:.4f}',f'{row["overall"]["perplexity"]:.2f}',
         f'{row["profiles"]["profile:strict"]["nll"]:.4f}',
         f'{row["profiles"]["profile:expanded_only"]["nll"]:.4f}'] for row in data['corpus_matrix']],
        'Четыре обучения с нуля, около 210 млн токенов на вариант. Общая validation: 10 200 341 токен, 1992 записи, 98 семейств.')
    architecture = table(['Вариант','Слои × ширина','Оптимизатор','NLL ↓','Top-1 ↑','48 ток., /10','96 ток., /10','Минуты'], [
        [row['label'],f'{row["config"]["model"]["n_layers"]} × {row["config"]["model"]["d_model"]}',
         'AdamW' if row['config']['optimizer']=='adamw' else 'Muon + AdamW',
         f'{row["validation"]["nll"]:.4f}',f'{row["validation"]["token_accuracy"]:.1%}',
         score(row,'early'),score(row,'full'),f'{row["endpoint"]["through_validation_minutes"]:.1f}']
        for row in data['architectures']],
        'По 420 млн токенов с нуля; одинаковые корпус, BPE, seed и batch. NLL/Top-1: 16 384 токена, 256 окон, 73 семейства. Средний балл: 23 начала × два слепых чтения, шкала ниже. Время — обучение с validation на RTX 4080 SUPER.')
    budgets = table(['Checkpoint','Обучено, млн ток.','NLL ↓','48 ток., /10','96 ток., /10'], [
        [row['label'],f'{row["checkpoint"]["tokens_seen"]/1e6:.2f}',f'{row["validation"]["nll"]:.4f}',
         score(row,'early'),score(row,'full')] for row in data['budgets']],
        'Повторное слепое сравнение 23 общих начал. Сравнивать попарно B630/D630 и A735/C735; A1050 — контроль большего бюджета. Пропущенные оценки исключены из исходных знаменателей.')
    pilots = table(['Метод','Дополнительно, млн ток.','NLL ↓','PPL ↓','48 ток., /10','96 ток., /10'], [
        [dict(D630='D630 · родитель',CE50='Обычный CE',Overlap50='Перекрытие 128 токенов',MTP50='Прогноз второго токена')[row['label']],
         '0' if row['label']=='D630' else '50',f'{row["validation"]["nll"]:.4f}',f'{row["validation"]["perplexity"]:.2f}',
         score(row,'early'),score(row,'full')]
        for row in data['pilots']],
        'Независимые продолжения D630; общая validation-панель 16 384 токена. Средний балл: 23 начала × два слепых чтения. Поздний MTP с весом 0,1 не проверяет MTP с самого начала обучения.')
    return dict(__MATRIX_TABLE__=matrix,__ARCHITECTURE_TABLE__=architecture,
                __BUDGET_TABLE__=budgets,__PILOTS_TABLE__=pilots)
