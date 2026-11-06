// Offline SVG chart of actual training observations. No external dependencies.
const chartNS = 'http://www.w3.org/2000/svg';
const metricNames = {nll: 'NLL (loss)', perplexity: 'Validation perplexity', lr: 'Learning rate',
    accuracy: 'Top-1 accuracy'};

function svgNode(tag, attributes, text) {
    const node = document.createElementNS(chartNS, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
}

function chartNumber(value, metric) {
    if (metric === 'accuracy') return (100*value).toFixed(2)+'%';
    return metric === 'lr' ? value.toExponential(1) : value.toFixed(metric === 'nll' ? 3 : 2);
}

function selectHistory() {
    const history = report.histories[Number(document.getElementById('history').value)];
    const metric = document.getElementById('history-metric');
    const previous = metric.value;
    metric.replaceChildren();
    for (const key of Object.keys(history.curves)) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = metricNames[key];
        metric.append(option);
    }
    metric.value = previous in history.curves ? previous : 'nll';
    drawHistory();
}

function drawHistory() {
    const history = report.histories[Number(document.getElementById('history').value)];
    const metric = document.getElementById('history-metric').value;
    const series = history.curves[metric];
    const all = series.flatMap(s => s.points);
    const width = 900, height = 340;
    const margin = {left: 82, right: 24, top: 42, bottom: 58};
    const xmax = Math.max(...all.map(p => p.x));
    let ymin = Math.min(...all.map(p => p.y)), ymax = Math.max(...all.map(p => p.y));
    const padding = Math.max((ymax - ymin) * .12, ['accuracy','lr'].includes(metric) ? 1e-8 : .015);
    ymin = metric === 'lr' ? 0 : Math.max(0, ymin - padding);
    ymax += padding;
    const x = value => margin.left + value / xmax * (width - margin.left - margin.right);
    const y = value => height - margin.bottom - (value - ymin) / (ymax - ymin) * (height - margin.top - margin.bottom);
    const chart = document.getElementById('history-chart');
    chart.replaceChildren();
    chart.setAttribute('viewBox', `0 0 ${width} ${height}`);
    chart.setAttribute('aria-label', `${history.label}. ${metricNames[metric]}`);
    chart.dataset.history = history.id;
    chart.dataset.metric = metric;
    for (let i = 0; i <= 4; i++) {
        const value = ymin + (ymax - ymin) * i / 4;
        chart.append(svgNode('line', {x1: margin.left, x2: width-margin.right, y1: y(value), y2: y(value), class: 'chart-grid'}));
        chart.append(svgNode('text', {x: margin.left-12, y: y(value)+5, 'text-anchor': 'end'}, chartNumber(value, metric)));
        const tokenValue = xmax * i / 4;
        chart.append(svgNode('text', {x: x(tokenValue), y: height-margin.bottom+24, 'text-anchor': 'middle'},
            (tokenValue/history.scale).toFixed(history.scale === 1e6 && xmax < 5e6 ? 2 : 1)));
    }
    const axisTitle = {perplexity:'Perplexity', nll:'NLL', lr:'LR', accuracy:'Top-1, %'};
    chart.append(svgNode('text', {x: margin.left, y: 20}, axisTitle[metric]));
    chart.append(svgNode('text', {x: width/2, y: height-8, 'text-anchor': 'middle'}, `С начала этапа, ${history.unit}`));
    for (const boundary of history.boundaries || []) {
        chart.append(svgNode('line', {x1: x(boundary.x), x2: x(boundary.x), y1: margin.top, y2: height-margin.bottom,
            stroke: '#a6afa7', 'stroke-dasharray': '3 4'}));
        chart.append(svgNode('text', {x: x(boundary.x)-6, y: 35, 'text-anchor': 'end', class: 'boundary-label'}, boundary.label));
    }
    const readout = document.getElementById('history-readout');
    readout.textContent = 'Точные значения — при наведении или фокусе на точке.';
    const legend = document.getElementById('history-legend');
    legend.replaceChildren();
    for (const item of series) {
        const label = document.createElement('span');
        label.textContent = item.label;
        label.style.borderColor = item.color;
        if (item.dashed) label.style.borderTopStyle = 'dashed';
        legend.append(label);
        if (item.connect) {
            chart.append(svgNode('polyline', {points: item.points.map(p => `${x(p.x)},${y(p.y)}`).join(' '),
                fill: 'none', stroke: item.color, 'stroke-width': 2.5, 'stroke-dasharray': item.dashed ? '7 5' : ''}));
        }
        for (const point of item.points) {
            const description = `${item.label}: ${chartNumber(point.y, metric)}; ${(point.x/history.scale).toFixed(3)} ${history.unit}`;
            const dot = svgNode('circle', {cx: x(point.x), cy: y(point.y), r: item.label.startsWith('Train') ? 3 : 5,
                fill: item.color, tabindex: 0, role: 'img', 'aria-label': description,
                'data-x': point.x, 'data-y': point.y, 'data-series': item.label});
            dot.append(svgNode('title', {}, description));
            for (const event of ['pointerenter', 'focus']) dot.addEventListener(event, () => {readout.textContent = description;});
            chart.append(dot);
        }
    }
    document.getElementById('history-description').textContent = history.description;
    document.getElementById('history-note').textContent = history.note;
    document.getElementById('history-source').href = '../../' + history.source;
    document.getElementById('history-source').textContent = 'Исходные измерения';
}

const historyGroups = new Map();
const historySelect = document.getElementById('history');
report.histories.forEach((history, index) => {
    if (!historyGroups.has(history.group)) {
        const group = document.createElement('optgroup');
        group.label = history.group;
        historySelect.append(group);
        historyGroups.set(history.group, group);
    }
    const option = document.createElement('option');
    option.value = index;
    option.textContent = history.label;
    historyGroups.get(history.group).append(option);
});
document.getElementById('history').addEventListener('change', selectHistory);
document.getElementById('history-metric').addEventListener('change', drawHistory);
for (const button of document.querySelectorAll('button[data-history]')) {
    button.addEventListener('click', () => {
        document.getElementById('history').value = report.histories.findIndex(r => r.id === button.dataset.history);
        selectHistory();
        document.getElementById('training-curves').scrollIntoView({behavior: 'smooth', block: 'start'});
    });
}
selectHistory();
