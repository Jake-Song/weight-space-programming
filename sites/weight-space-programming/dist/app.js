'use strict';
const conditions = [
  {id:'base',name:'원본 · thinking on',short:'원본 · thinking on',accuracy:8,cut:92,tokens:30587.39,loop:88,color:'#77869b',description:'가중치를 수정하지 않은 기준 모델입니다. thinking을 켜고 동일한 생성 설정으로 평가했습니다.',note:'92개 응답이 최대 생성 길이 32,768토큰에 도달했습니다.'},
  {id:'layer15',name:'레이어 15만 수정',short:'레이어 15만 수정',accuracy:41,cut:42,tokens:22562.53,loop:22,color:'#77bda7',description:'공개 diff 중 15번 레이어의 MLP 수정만 적용했습니다. 64개 출력 열의 가중치가 대상입니다.',note:'원본 대비 +33%p. 문제 단위 재표집 95% 구간: +23~+43%p.'},
  {id:'layer19',name:'레이어 19만 수정',short:'레이어 19만 수정',accuracy:31,cut:66,tokens:26209.06,loop:62,color:'#659e91',description:'공개 diff 중 19번 레이어의 MLP 수정만 적용했습니다. 15번 레이어는 원본 상태입니다.',note:'원본 대비 +23%p. 문제 단위 재표집 95% 구간: +14~+32%p.'},
  {id:'both',name:'두 레이어 수정',short:'두 레이어 수정',accuracy:57,cut:26,tokens:20638.26,loop:7,color:'#a4f0d2',description:'블로그의 최종 수정값을 두 레이어 모두에 적용했습니다. 수정 텐서의 해시 일치를 검증했습니다.',note:'원본 대비 +49%p. 문제 단위 재표집 95% 구간: +39~+59%p.'},
  {id:'nothink',name:'원본 · thinking off',short:'원본 · thinking off',accuracy:51,cut:12,tokens:8245.21,loop:5,color:'#f0bb88',description:'원본 가중치를 사용하면서 thinking 옵션만 껐습니다. 샘플링과 최대 생성 길이는 동일합니다.',note:'수정본의 약 40% 토큰으로 51% 정답률. 수정본과 정확도 차이는 불확실합니다.'},
  {id:'random',name:'무작위 방향 수정',short:'무작위 방향 수정',accuracy:11,cut:89,tokens:29672.1,loop:86,color:'#768097',description:'같은 열에서 수정 크기를 맞추고 방향과 부호를 무작위로 바꿨습니다. 무작위 시드 1개를 사용했습니다.',note:'원본 대비 +3%p. 문제 단위 재표집 95% 구간: −3~+9%p.'}
];
const format = new Intl.NumberFormat('ko-KR',{maximumFractionDigits:0});
let metric='accuracy', selected='both';
const bars=document.getElementById('bars');
const metricInfo={accuracy:{max:100,label:'정답률 (%)',hint:'높을수록 좋음 ↑',unit:'%'},cut:{max:100,label:'토큰 한도 도달률 (%)',hint:'낮을수록 좋음 ↓',unit:'%'},tokens:{max:32768,label:'평균 생성 토큰',hint:'같은 정확도라면 낮을수록 효율적 ↓',unit:''}};
function renderBars(){
  const info=metricInfo[metric];
  document.getElementById('metric-hint').textContent=info.hint;
  document.getElementById('scale-label').textContent=info.label;
  document.getElementById('scale-max').textContent=format.format(info.max)+info.unit;
  bars.replaceChildren();
  conditions.forEach(c=>{
    const button=document.createElement('button');button.className='bar-row'+(selected===c.id?' selected':'');
    button.type='button';button.setAttribute('aria-pressed',String(selected===c.id));button.setAttribute('aria-label',`${c.name}, ${info.label} ${format.format(c[metric])}${info.unit}, 상세 보기`);
    const name=document.createElement('span');name.className='bar-name';name.textContent=c.short;
    const track=document.createElement('span');track.className='bar-track';track.setAttribute('aria-hidden','true');
    const fill=document.createElement('span');fill.className='bar-fill';fill.style.display='block';fill.style.setProperty('--bar-color',c.color);fill.style.width=`${c[metric]/info.max*100}%`;track.append(fill);
    const value=document.createElement('span');value.className='bar-value';value.textContent=format.format(c[metric])+info.unit;
    button.append(name,track,value);button.addEventListener('click',()=>{selected=c.id;updateSelection();});button.dataset.condition=c.id;bars.append(button);
  });
}
function updateSelection(){
  const c=conditions.find(c=>c.id===selected);
  for(const b of bars.children){const active=b.dataset.condition===selected;b.classList.toggle('selected',active);b.setAttribute('aria-pressed',String(active));}
  document.getElementById('detail-title').textContent=c.name;
  document.getElementById('detail-description').textContent=c.description;
  const number=document.getElementById('detail-accuracy');number.replaceChildren(document.createTextNode(c.accuracy));const unit=document.createElement('span');unit.textContent='%';number.append(unit);
  document.getElementById('detail-correct').textContent=`${c.accuracy} / 100`;
  document.getElementById('detail-cut').textContent=c.cut+'%';
  document.getElementById('detail-tokens').textContent=format.format(c.tokens);
  document.getElementById('detail-loop').textContent=c.loop+'%';
  document.getElementById('detail-note').textContent=c.note;
}
document.querySelectorAll('[data-metric]').forEach(b=>b.addEventListener('click',()=>{
  metric=b.dataset.metric;document.querySelectorAll('[data-metric]').forEach(x=>{const active=x===b;x.classList.toggle('active',active);x.setAttribute('aria-pressed',String(active));});renderBars();
}));
const from=document.getElementById('from-condition'),to=document.getElementById('to-condition');
for(const select of [from,to]) conditions.forEach(c=>{const option=document.createElement('option');option.value=c.id;option.textContent=c.name;select.append(option);});
from.value='nothink';to.value='both';
const intervals={base:{both:[39,59],layer15:[23,43],layer19:[14,32],nothink:[33,53],random:[-3,9]},layer15:{both:[8,24]},layer19:{both:[17,35]},nothink:{both:[-3,15]},random:{both:[37,56]}};
const signed=n=>n>0?'+'+n:n<0?'−'+Math.abs(n):'0';
function comparison(){
  const a=conditions.find(c=>c.id===from.value),b=conditions.find(c=>c.id===to.value);
  document.getElementById('delta-accuracy').textContent=signed(b.accuracy-a.accuracy)+'%p';
  document.getElementById('delta-tokens').textContent=(b.tokens/a.tokens).toFixed(2)+'×';
  document.getElementById('delta-cut').textContent=signed(b.cut-a.cut)+'%p';
  let interval=intervals[a.id]?.[b.id];if(!interval&&intervals[b.id]?.[a.id]){const inv=intervals[b.id][a.id];interval=[-inv[1],-inv[0]];}
  let note=a.id===b.id?'같은 조건을 선택했습니다. 다른 조건을 선택하면 차이를 볼 수 있습니다.':`토큰 비율은 ‘${b.name} ÷ ${a.name}’입니다. `;
  if(interval) note+=`정답률 차이의 탐색적 95% 구간: ${signed(interval[0])}~${signed(interval[1])}%p. `+(interval[0]<=0&&interval[1]>=0?'0을 포함해 정확도 차이를 확정하기 어렵습니다.':'문제 단위 재표집 구간이며 생성 시드 변동은 포함하지 않습니다.');
  else if(a.id!==b.id) note+='문제당 1회 생성한 탐색 결과입니다. 이 두 조건의 차이에 대한 구간은 별도로 계산하지 않았습니다.';
  document.getElementById('comparison-note').textContent=note;
}
from.addEventListener('change',comparison);to.addEventListener('change',comparison);
const tbody=document.getElementById('table-body');conditions.forEach(c=>{const row=document.createElement('tr');[c.name,c.accuracy+'%',c.cut+'%',format.format(c.tokens),c.loop+'%'].forEach((v,i)=>{const td=document.createElement(i===0?'th':'td');if(i===0)td.scope='row';td.textContent=v;row.append(td);});tbody.append(row);});
renderBars();updateSelection();comparison();
