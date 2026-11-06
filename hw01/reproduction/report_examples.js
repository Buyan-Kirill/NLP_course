function options(id, labels) {
    const select = document.getElementById(id);
    labels.forEach((label, i) => {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = label;
        select.append(option);
    });
}

function verse(id, text) {
    const node = document.getElementById(id);
    node.replaceChildren();
    if (text === null) { node.textContent = 'Нет допустимого продолжения'; return; }
    const match = [...text.matchAll(/[А-Яа-яЁё]+/g)].pop();
    if (!match) { node.textContent = text; return; }
    const mark = document.createElement('mark');
    mark.textContent = match[0];
    node.append(document.createTextNode(text.slice(0, match.index)), mark,
                document.createTextNode(text.slice(match.index + match[0].length)));
}

function showRatings(id, mode) {
    const target = document.getElementById(id);
    target.replaceChildren();
    const table = document.createElement('table');
    const header = document.createElement('tr');
    for (const title of ['Оценщик', 'Связь', 'Мысль', 'Логика', 'Речь', 'Манера', 'Σ /10']) {
        const cell = document.createElement('th');
        cell.textContent = title;
        header.append(cell);
    }
    table.append(header);
    mode.ratings.forEach((reader, index) => {
        const row = document.createElement('tr');
        for (const value of [`Агент ${index+1}`, ...['relevance','progress','logic','naturalness','style'].map(k => reader.scores[k]), reader.total]) {
            const cell = document.createElement('td');
            cell.textContent = value;
            row.append(cell);
        }
        table.append(row);
    });
    target.append(table);
}

function showExample() {
    const row = report.examples[Number(document.getElementById('example').value)];
    verse('prompt', row.prefix);
    for (const mode of ['free','constrained']) {
        verse(mode, row[mode].text);
        const scores = [...new Set(row[mode].ratings.map(r => r.total))].sort((a,b) => a-b);
        document.getElementById(mode+'-score').textContent = scores.join('–')+'/10';
        showRatings(mode+'-ratings', row[mode]);
    }
    const source = document.getElementById('example-source');
    source.href = new URL(row.source, report.repository).href;
    source.textContent = 'Протокол генерации (JSON) · SHA '+row.checkpoint_sha256.slice(0,12)+'…';
}

function showStages() {
    const index = Number(document.getElementById('stage-case').value);
    document.getElementById('stage-prompt').textContent = report.stages[0].examples[index].prompt;
    const target = document.getElementById('stage-examples');
    target.replaceChildren();
    for (const stage of report.stages) {
        const card = document.createElement('article');
        card.className = 'card';
        const title = document.createElement('h3');
        title.textContent = stage.label;
        const text = document.createElement('pre');
        text.className = 'stage-text';
        text.textContent = stage.examples[index].text;
        const source = document.createElement('a');
        source.className = 'source';
        source.href = new URL(stage.examples[index].source, report.repository).href;
        source.textContent = 'Протокол генерации (JSON)';
        card.append(title, text, source);
        target.append(card);
    }
}

options('example', report.examples.map(r => r.prefix));
options('stage-case', report.stages[0].examples.map(r => r.prompt.trim()));
document.getElementById('example').addEventListener('change', showExample);
document.getElementById('stage-case').addEventListener('change', showStages);
showExample();
showStages();
