'use strict';
(async function () {
  const response = await fetch('diagnostics.json');
  if (!response.ok) throw new Error('진단 데이터 로딩 실패');
  const data = await response.json();
  const colors = {10: '#a4f0d2', 19: '#b3a0ff', 23: '#f0bb88'};
  const tabs = document.getElementById('candidate-tabs');
  const chart = document.getElementById('ratio-chart');
  const verdict = document.getElementById('candidate-verdict');
  const make = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  function select(layer, notify = true) {
    const candidate = data.candidates.find(c => c.layer === layer);
    tabs.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.candidate) === layer)));
    chart.replaceChildren();
    chart.style.setProperty('--candidate-color', colors[layer]);
    candidate.ratios.forEach((ratio, index) => {
      const row = make('div', 'ratio-row');
      const track = make('div', 'ratio-track');
      track.setAttribute('aria-hidden', 'true');
      const bar = make('span', 'ratio-fill');
      bar.style.width = `${ratio / 2 * 100}%`;
      track.append(bar, make('span', 'ratio-baseline'), make('span', 'ratio-threshold'));
      row.append(make('span', 'ratio-label', data.problems[index].label), track, make('strong', 'ratio-value', `${ratio.toFixed(4)}×`));
      chart.append(row);
    });
    verdict.replaceChildren();
    const badge = make('span', candidate.criterion_met ? 'verdict-badge passed' : 'verdict-badge', candidate.criterion_met ? '기준 충족' : '기준 미충족');
    const details = make('div');
    details.append(make('strong', '', `Layer ${layer} · 채널 ${candidate.channels.join(', ')}`), make('p', '', `1× 초과 ${candidate.greater_than_1}/4쌍 · 1.25× 이상 ${candidate.at_least_1_25}/4쌍`));
    verdict.append(badge, details);
    if (notify) window.dispatchEvent(new CustomEvent('diagnostic-select', {detail: {layer}}));
  }
  data.candidates.forEach(c => {
    const button = make('button', '', `Layer ${c.layer}`);
    button.type = 'button'; button.dataset.candidate = c.layer;
    button.style.setProperty('--candidate-color', colors[c.layer]);
    button.addEventListener('click', () => select(c.layer));
    tabs.append(button);
    const row = make('tr');
    const heading = make('th', '', `L${c.layer} · ${c.channels.join(', ')}`); heading.scope = 'row'; row.append(heading);
    c.ratios.forEach(r => row.append(make('td', '', r.toFixed(4))));
    row.append(make('td', c.criterion_met ? 'mint-text' : '', c.criterion_met ? '충족' : '미충족'));
    document.getElementById('candidate-table').append(row);
  });
  data.steps.forEach(step => {
    const item = make('details', 'step-item');
    const summary = make('summary');
    summary.append(make('span', 'step-number', String(step.number).padStart(2, '0')), make('span', 'step-title', step.title), make('span', 'step-kind', step.kind), make('span', `step-state${['오류', '부분 분석'].includes(step.status) ? ' needs-review' : ''}`, step.status));
    item.append(summary, make('p', '', step.description));
    document.getElementById('step-list').append(item);
  });
  select(10, false);
  document.getElementById('diagnostic-status').hidden = true;
  document.getElementById('diagnostic').dataset.ready = 'true';
})().catch(error => {
  document.getElementById('diagnostic-status').textContent = '진단 데이터를 불러오지 못했습니다. 새로고침하거나 결과 보고서를 확인해 주세요.';
  document.getElementById('diagnostic').dataset.ready = 'error';
  console.error(error);
});
