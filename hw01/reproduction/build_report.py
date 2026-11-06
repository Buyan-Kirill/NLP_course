"""Build the offline report exclusively from saved measurements and raw outputs."""
import hashlib
import html
import json
import math
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit

from reproduction import report_charts, report_data_quality, report_experiments, report_provenance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/report"
REPOSITORY = "https://github.com/Buyan-Kirill/NLP_course/blob/main/hw01/"

SOURCE_TITLES = {
    "reproduction/evidence/inference22_config.json": "Конфигурация записанного примера22",
    "artifacts/releases/poetry17/manifest.json": "Выбранные веса и цепочка обучения",
    "data/prepared_v7/expanded/manifest.json": "Состав прозаического корпуса",
    "data/prepared_v7/poetry_supplement/manifest.json": "Состав поэтического корпуса",
    "audit/pretrain11/final_automatic_review.json": "Сравнение вариантов претрейна",
    "audit/continuation19/teacher_forced/result.json": "Предсказание эталонных продолжений до и после SFT",
    "audit/continuation19/dev_reading/analysis/summary.json": "Оценки продолжений на открытой выборке",
    "audit/continuation19/final_reading/analysis/summary.json": "Заключительные оценки двух читателей",
    "audit/continuation19/inference/parent_dev/completed.json": "Ответы одной модели с поиском и без поиска",
    "audit/rhyme17/final17/reading/analysis/summary.md": "Результаты раннего сравнения рифмовки",
    "docs/series17_summary.md": "Обзор экспериментов с окончаниями и рифмой",
    "docs/series19_summary.md": "Обзор SFT продолжений",
    "docs/pretrain11_results.md": "Результаты изменения LR, корпуса и шума",
    "audit/continuation19/evaluation/dev_pool.json": "Словари рифм для показанных примеров",
    "reproduction/evidence/lr50_metrics.jsonl": "Журнал validation при завершении претрейна",
    "audit/poetry16/analysis_3231558/summary.json": "Сравнение проходов поэзии и размеров batch",
    "audit/poetry08/lr50_diagnostic13/parent_validation.json": "Поэтическая validation до адаптации",
    "audit/poetry16/plan_trial/planned/400000_validation.json": "Поэтическая validation выбранных весов",
    "audit/poetry08/lr50_diagnostic13/parent_poetry.json": "Свободный текст до поэтической адаптации",
    "audit/poetry08/lr50_diagnostic13/pure_1615779_poetry.json": "Свободный текст после первого прохода поэзии",
    "audit/poetry16/batch4k/3231558_poetry.json": "Свободный текст после второго прохода поэзии",
    "audit/poetry16/plan_trial/planned/400000_poetry.json": "Свободный текст выбранных весов",
    "reproduction/evidence/stress_fix22_report.json": "Пример в начале отчёта после исправления ударений",
    "audit/finalize18/final_metrics_preparation/result.json": "Validation инструкционной ветви SFT",
    "audit/finalize18/inference/chat12/evidence/analysis/summary.json": "Оценка свободных ответов после SFT",
}
SOURCE_TITLES.update(report_charts.SOURCES)
SOURCE_TITLES.update(report_provenance.SOURCES)
SOURCE_TITLES.update(report_data_quality.SOURCES)
SOURCE_TITLES.update(report_experiments.SOURCES)


def read(name):
    return json.loads((ROOT / name).read_text("utf-8-sig"))


def portable_links(document):
    """Keep in-page anchors local; open repository evidence from any HTML location."""
    def replace(match):
        href = html.unescape(match[1])
        parsed = urlsplit(href)
        if not href or href.startswith('#') or parsed.scheme or parsed.netloc:
            return match[0]
        target = (OUT / unquote(parsed.path)).resolve()
        relative = target.relative_to(ROOT).as_posix()
        url = REPOSITORY + quote(relative, safe='/')
        if parsed.query:
            url += '?' + parsed.query
        if parsed.fragment:
            url += '#' + parsed.fragment
        return 'href="' + html.escape(url, quote=True) + '"'
    return re.sub(r'href="([^"]*)"', replace, document)


def report_data():
    sources = [
        "artifacts/releases/poetry17/manifest.json",
        "data/prepared_v7/expanded/manifest.json",
        "data/prepared_v7/poetry_supplement/manifest.json",
        "audit/pretrain11/final_automatic_review.json",
        "audit/continuation19/teacher_forced/result.json",
        "audit/continuation19/dev_reading/analysis/summary.json",
        "audit/continuation19/final_reading/analysis/summary.json",
        "audit/continuation19/inference/parent_dev/completed.json",
    ]
    pretrain = read(sources[3])
    comparisons = []
    for label in ("CE50", "LR50", "Noise50", "Mix50"):
        model = pretrain["main"]["models"][label]
        val = model["teacher_forced"]["validation"]
        comparisons.append({"label": label, "nll": val["nll"], "accuracy": val["token_accuracy"],
            "perplexity": val["perplexity"],
            "distinct2": model["generation"]["sampling"]["validation"]["distinct2"]["mean"],
            "repeat4": model["generation"]["sampling"]["validation"]["repeated4"]["mean"],
            "n_cases": val["n_cases"], "families": val["n_families"], "targets": val["targets"],
            "source": sources[3]})
    final = read(sources[6])
    pair_scores = []
    for reader, data in final["readers"].items():
        for mode in ("free", "constrained"):
            row = data["conditions"]["pair/planned400k/" + mode]
            pair_scores.append({"reader": reader, "mode": mode, "n": row["N"],
                "joint": row["flags"]["joint"]["true"], "rhyme": row["flags"]["rhyme"]["true"],
                "meaning": row["flags"]["meaning_acceptable"]["true"], "score": row["mean_total"],
                "source": sources[6]})
    teacher = read(sources[4])
    native = [{"label": row["label"], **row["report"]["metrics"]["overall"]} for row in teacher["models"]]
    raw = read(sources[7])["samples"]
    pool_source = 'audit/continuation19/evaluation/dev_pool.json'
    sources.append(pool_source)
    pools = {r['id']: r for r in read(pool_source)['results']}
    examples = []
    # A fixed spread of source order, including the known failure; no text editing.
    for index in range(6):
        sample = raw[index]
        result = sample.get("constrained")
        examples.append({"index": index, "prefix": sample["case"]["prefix"],
                         "free": sample["free"]["text"],
                         "pool": [r['word'] for r in pools[sample['case']['id']]['candidates']],
                         "text": result.get("text") if result else None,
                         "source": sources[7], "scope": "open dev; not final holdout"})
    prose = read(sources[1])["splits"]["train"]["prose"]
    poetry = read(sources[2])["splits"]["train"]["poetry"]
    poetry_source = 'audit/poetry16/analysis_3231558/summary.json'
    base_source = 'audit/poetry08/lr50_diagnostic13/parent_validation.json'
    sources.extend([poetry_source, base_source])
    base = read(base_source)['domains']['poetry']['metrics']['overall']
    poetry_rows = [dict(label='После претрейна', **base, source=base_source)]
    for key, label in [('PureP', 'Поэзия: первый проход'),
                       ('batch16k', 'Второй проход, batch 16 384'),
                       ('batch4k', 'Второй проход, batch 4096')]:
        metrics = read(poetry_source)['models'][key]['validation']['poetry']['slices']['overall']
        assert metrics['targets'] == base['targets'] and metrics['families'] == base['families']
        poetry_rows.append(dict(label=label, **metrics, source=poetry_source))
    planned_source = 'audit/poetry16/plan_trial/planned/400000_validation.json'
    sources.append(planned_source)
    metrics = read(planned_source)['domains']['poetry']['metrics']['overall']
    assert metrics['targets'] == base['targets'] and metrics['families'] == base['families']
    poetry_rows.append(dict(label='Поэзия + подсказка окончания (выбрано)', **metrics, source=planned_source))
    stage_sources = [
        ('После претрейна', 'audit/poetry08/lr50_diagnostic13/parent_poetry.json'),
        ('Первый проход поэзии', 'audit/poetry08/lr50_diagnostic13/pure_1615779_poetry.json'),
        ('Второй проход, batch 4096', 'audit/poetry16/batch4k/3231558_poetry.json'),
        ('Поэзия + подсказка окончания', 'audit/poetry16/plan_trial/planned/400000_poetry.json'),
    ]
    stages = []
    for label, name in stage_sources:
        sources.append(name)
        rows = read(name)['samples']
        # Fixed source positions and seed, shared prompts; no editorial rewriting.
        selected = [dict(id=rows[i]['case']['id'], prompt=rows[i]['case']['prompt'],
                         text=rows[i]['text'], seed=rows[i]['seed'], source=name) for i in (0, 4, 8)]
        if stages:
            assert [(r['id'], r['prompt'], r['seed']) for r in selected] == [
                (r['id'], r['prompt'], r['seed']) for r in stages[0]['examples']]
        stages.append(dict(label=label, examples=selected))
    for row in [*comparisons, *poetry_rows, *native]:
        assert math.isclose(row['perplexity'], math.exp(row['nll']), rel_tol=1e-6)
    hero_source = "reproduction/evidence/stress_fix22_report.json"
    sources.append(hero_source)
    hero = read(hero_source)['samples'][2]
    assert hero['case']['first_line'] == 'В саду под вечер свет горит,' and hero['status'] == 'selected'
    chat_source = "audit/finalize18/final_metrics_preparation/result.json"
    sources.append(chat_source)
    chat = []
    for row in read(chat_source)['models']:
        if row['label'] in ('poetry4k', 'sft183596'):
            metrics = row['phases']['sft_token_partitions']['metrics']['assistant_content']
            chat.append(dict(label=row['label'], nll=metrics['nll'],
                perplexity=math.exp(metrics['nll']), accuracy=metrics['top1'],
                tokens=metrics['targets'], source=chat_source))
    assert len(chat) == 2 and all(r['tokens'] == 3321 for r in chat)
    chat_reading_source = 'audit/finalize18/inference/chat12/evidence/analysis/summary.json'
    sources.append(chat_reading_source)
    chat_reading = []
    for reader, conditions in read(chat_reading_source)['summaries'].items():
        row = conditions['chat/sft183596/ordinary']
        chat_reading.append(dict(reader=reader, answers=row['attempts'], situations=row['families'],
            meaning=row['flags']['meaning_acceptable']['true'], joint=row['flags']['joint']['true']))
    assert all((r['answers'], r['situations'], r['meaning'], r['joint']) == (14, 12, 1, 0) for r in chat_reading)
    sources.append("reproduction/evidence/inference22_config.json")
    sources.extend(name for name in [*report_charts.SOURCES, *report_provenance.SOURCES,
                                    *report_data_quality.SOURCES, *report_experiments.SOURCES] if name not in sources)
    data = {"as_of": "2026-11-06", "parameters": 68440576, "repository": REPOSITORY,
            "prose": prose, "poetry": poetry, "pretrain": comparisons,
            "poetry_metrics": poetry_rows, "stages": stages,
            "final": pair_scores, "native": native, "open_examples": examples,
            "hero": dict(input=hero['case']['first_line'], output=hero['selected']['text'].strip(), source=hero_source),
            "chat_metrics": chat, "chat_reading": chat_reading,
            "sources": [{"path": name, "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
                        for name in sources]}
    data['examples'] = report_provenance.final_examples()
    data['showcase'] = report_provenance.showcase()
    data['data_quality'] = report_data_quality.quality_data()
    data['experiments'] = report_experiments.experiment_data()
    data['histories'] = report_charts.histories(poetry_rows, native)
    data['provenance'] = report_provenance.verify_examples(data)
    return data


def main():
    data = report_data()
    OUT.mkdir(parents=True, exist_ok=True)
    template = (ROOT / "reproduction/report_template.html").read_text("utf8")
    labels = dict(CE50='CE, контроль', LR50='LR ×0,5', Noise50='Шум 5%', Mix50='Смесь 70/30')
    pre_rows = "".join(f'<tr><td>{labels[r["label"]]}</td><td>{r["nll"]:.4f}</td><td>{r["perplexity"]:.2f}</td><td>{r["accuracy"]:.1%}</td><td>{r["distinct2"]:.1%}</td><td>{r["repeat4"]:.2%}</td></tr>' for r in data["pretrain"])
    final_rows = []
    for mode, label in [('free', 'Свободная генерация'), ('constrained', 'Та же модель + поиск')]:
        rows = [r for r in data['final'] if r['mode'] == mode]
        assert len(rows) == 2 and rows[0]['n'] == rows[1]['n']
        def count(key):
            values = sorted({r[key] for r in rows})
            return ('–'.join(map(str, values))) + '/' + str(rows[0]['n'])
        scores = sorted(r['score'] for r in rows)
        final_rows.append(f'<tr><td>{label}</td><td>{count("rhyme")}</td><td>{count("joint")}</td><td>{scores[0]:.2f}–{scores[-1]:.2f}</td></tr>')
    source_rows = "".join(
        f'<li><a href="../../{html.escape(r["path"])}">{html.escape(SOURCE_TITLES[r["path"]])}</a>'
        f'<span class="source-meta">{Path(r["path"]).suffix[1:].upper()} · SHA256 {r["sha256"][:12]}…</span></li>'
        for r in data["sources"])
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    replacements = {"__DATA__": payload, "__PRE_ROWS__": pre_rows, "__FINAL_ROWS__": ''.join(final_rows),
                    "__SOURCES__": source_rows,
                    "__HERO_INPUT__": html.escape(data['hero']['input']),
                    "__HERO_OUTPUT__": html.escape(data['hero']['output'])}
    replacements['__POETRY_ROWS__'] = ''.join(f'<tr><td>{html.escape(r["label"])}</td><td>{r["nll"]:.4f}</td><td>{r["perplexity"]:.2f}</td></tr>' for r in data['poetry_metrics'])
    replacements['__NATIVE_ROWS__'] = ''.join(f'<tr><td>{"До SFT" if r["label"] == "planned400k" else "После SFT"}</td><td>{r["nll"]:.4f}</td><td>{r["perplexity"]:.2f}</td></tr>' for r in data['native'])
    replacements['__CHAT_ROWS__'] = ''.join(f'<tr><td>{"Поэзия, batch 4096" if r["label"] == "poetry4k" else "После chat-SFT"}</td><td>{r["nll"]:.4f}</td><td>{r["perplexity"]:.2f}</td><td>{r["accuracy"]:.1%}</td></tr>' for r in data['chat_metrics'])
    replacements['__SHOWCASE__'] = ''.join(
        '<article class="card"><span class="tag">Сохранённая генерация</span>'
        f'<pre>{html.escape(row["prefix"])}\n{html.escape(row["output"].rstrip())}</pre>'
        '</article>'
        for row in data['showcase'][1:])
    def integer(value):
        return f'{value:,}'.replace(',', ' ')
    replacements['__CORPUS_ROWS__'] = ''.join(
        f'<tr><td>{label}</td><td>{integer(row["words"])}</td><td>{integer(row["word_types"])}</td>'
        f'<td>{row["word_frequency_types"]["1"]/row["word_types"]:.1%}</td>'
        f'<td>{row["document_words"]["p50"]} / {row["document_words"]["p95"]}</td>'
        f'<td>{row["script_letters"]["latin_share_of_cyrillic_and_latin"]:.3%}</td></tr>'
        for domain, label in [('prose','Проза'),('poetry','Поэзия')]
        for row in [data['data_quality']['corpus']['domains'][domain]])
    replacements['__WORD_HEAD__'] = ''.join(
        '<p><b>'+label+':</b> '+html.escape(', '.join(
            f'{r["word"]} ({integer(r["count"])})' for r in data['data_quality']['corpus']['domains'][domain]['top20'][:8]))+'</p>'
        for domain,label in [('prose','Проза'),('poetry','Поэзия')])
    replacements['__TOKENIZER_ROWS__'] = ''.join(
        f'<tr><td>{integer(row["size"])}</td><td>{"Претрейн + SFT" if row["mixture"] == "mixed" else "Претрейн"}'
        f'{" · выбрано" if row["selected"] else ""}</td><td>{row["tokens_per_word"]:.3f}</td>'
        f'<td>{row["rare_types"]}</td><td>{row["unused_types"]}</td></tr>'
        for row in data['data_quality']['tokenizers'])
    replacements['__TOKENIZER_DOMAIN_ROWS__'] = ''.join(
        f'<tr><td>{"Исходный 8k · выбрано" if row["name"] == "baseline_v2_8192" else "Новый " + str(row["size"]//1024) + "k"}</td><td>{row["prose"]:.3f}</td>'
        f'<td>{row["poetry"]:.3f}</td><td>{row["sft"]:.3f}</td></tr>'
        for row in data['data_quality']['tokenizers_by_domain'])
    replacements['__CHART_SCRIPT__'] = (ROOT/'reproduction/report_charts.js').read_text('utf8')
    for key, stage in [('PRETRAIN','pretrain'),('POETRY','poetry'),('SFT','sft')]:
        replacements['__'+key+'_CHART__'] = training_chart(stage)
    replacements.update(report_experiments.render_tables(data['experiments']))
    replacements['__EXAMPLE_SCRIPT__'] = (ROOT/'reproduction/report_examples.js').read_text('utf8')
    for key, value in replacements.items():
        template = template.replace(key, value)
    template = portable_links(template)
    (OUT / "index.html").write_text(template, encoding="utf8", newline="\n")
    (OUT / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8", newline="\n")
    (OUT / "sources.json").write_text(json.dumps(data["sources"], ensure_ascii=False, indent=2) + "\n", encoding="utf8", newline="\n")
    (OUT / "generation_sources.json").write_text(json.dumps(data['provenance'], ensure_ascii=False, indent=2) + "\n", encoding='utf8', newline="\n")
    print(f"Report: {OUT / 'index.html'}; {len(data['sources'])} inspected sources")


def training_chart(stage):
    return f'''<div class="card training-panel" data-training-stage="{stage}">
<h3>Динамика обучения</h3>
<div class="toolbar"><label for="{stage}-metric">Метрика</label><select id="{stage}-metric" data-role="metric"></select></div>
<p class="caption" data-role="description"></p><div class="chart-legend" data-role="legend"></div>
<div class="chart-scroll"><svg class="training-chart" data-role="chart" role="img"></svg></div>
<p class="caption" data-role="readout" aria-live="polite"></p><p class="caption" data-role="note"></p>
<p class="source" data-role="sources"></p></div>'''


if __name__ == "__main__":
    main()
