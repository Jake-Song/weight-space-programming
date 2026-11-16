import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createModel, LAYER_GAP } from './model-geometry.js';

const $=id=>document.getElementById(id);
const viewport=$('viewport');
const buttons=[];
for(let i=0;i<24;i++){
  const button=document.createElement('button');
  button.type='button';button.textContent=String(i).padStart(2,'0');button.dataset.layer=i;
  button.setAttribute('aria-label',`레이어 ${i}${[15,19].includes(i)?', MLP 수정됨':', 수정 없음'}`);
  button.setAttribute('aria-pressed','false');button.disabled=true;
  $('layer-buttons').append(button);buttons.push(button);
}

async function init(){
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.4;
  const canvas=renderer.domElement;canvas.setAttribute('aria-label','Qwen3.5 모델의 3D 구조. 옆의 레이어 선택 버튼으로도 탐색할 수 있습니다.');
  viewport.append(canvas);
  const scene=new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xc4ddff,0x293647,3));
  for(const [pos,color,intensity] of [[[5,16,10],0xd6e7ff,4],[[-8,9,3],0xc3b3ff,2],[[4,12,-8],0x9affce,3]]){
    const light=new THREE.DirectionalLight(color,intensity);light.position.set(...pos);scene.add(light);
  }
  const camera=new THREE.OrthographicCamera(-10,10,6,-6,.1,200);
  const controls=new OrbitControls(camera,canvas);
  // Diagram zoom changes scale only: neither pointer position nor pinch translation
  // can move the orbit target. Immediate rotation also makes reset deterministic.
  controls.enableDamping=false;controls.enablePan=false;controls.zoomToCursor=false;
  controls.minZoom=.25;controls.maxZoom=8;
  controls.maxPolarAngle=Math.PI*.92;controls.autoRotateSpeed=.65;
  let dirty=true,visible=true,selected=null,isolate=false,spacing=1,model=null;
  let fitHalfWidth=10,fitHalfHeight=6;
  const layers=new Map();
  const originalPositions=new Map();
  controls.addEventListener('change',()=>{dirty=true;$('zoom-readout').textContent=Math.round(camera.zoom*100)+'%';});
  const metadata=await fetch('assets/model-metadata.json?v=js-1').then(r=>{if(!r.ok)throw new Error('Metadata failed');return r.json();});
  model=createModel(metadata);scene.add(model);
  model.traverse(obj=>{
    if(Number.isInteger(obj.userData.layerIndex))layers.set(obj.userData.layerIndex,obj);
    if(Number.isInteger(obj.userData.layerIndex)||obj.userData.stage==='output')originalPositions.set(obj,obj.position.clone());
  });
  if(layers.size!==24)throw new Error('Expected 24 generated layer objects');
  const edits=new Map(metadata.edits.map(e=>[e.layer,e]));

  function fitView(object){
    model.updateMatrixWorld(true);
    const box=new THREE.Box3().setFromObject(object);
    box.getCenter(controls.target);
    controls.autoRotate=false;$('view-rotate').setAttribute('aria-pressed','false');
    camera.position.copy(controls.target).add(new THREE.Vector3(.22,.48,1).normalize().multiplyScalar(40));
    camera.lookAt(controls.target);camera.zoom=1;controls.update();
    const inverse=camera.quaternion.clone().invert();
    fitHalfWidth=0;fitHalfHeight=0;
    for(const x of [box.min.x,box.max.x])for(const y of [box.min.y,box.max.y])for(const z of [box.min.z,box.max.z]){
      const point=new THREE.Vector3(x,y,z).sub(controls.target).applyQuaternion(inverse);
      fitHalfWidth=Math.max(fitHalfWidth,Math.abs(point.x)*1.2);
      fitHalfHeight=Math.max(fitHalfHeight,Math.abs(point.y)*1.2);
    }
    resize();dirty=true;
  }
  function overview(){
    selected=null;updateInspector();fitView(model);
  }
  function focus(index){
    selected=index;
    if(isolate&&!edits.has(index)){isolate=false;applyVisibility();}
    updateInspector();fitView(layers.get(index));
  }
  function updateInspector(){
    for(const b of buttons)b.setAttribute('aria-pressed',String(Number(b.dataset.layer)===selected));
    if(selected===null){
      $('inspector-title').textContent='두 곳의 작은 수정.';
      $('inspector-title').style.color='';
      $('inspector-summary').textContent='전체 모델을 다시 학습하지 않고, 두 레이어의 MLP down_proj 가중치만 바꿨습니다.';
      $('fact-a-label').textContent='변경된 가중치 값';$('fact-a').textContent='130,188';
      $('fact-b-label').textContent='수정한 열';$('fact-b').textContent='64 × 2';
      $('fact-c-label').textContent='대상 모듈';$('fact-c').textContent='MLP · down_proj';
      $('inspector-note').textContent='작은 발광 블록 하나는 수정한 열 하나를 뜻합니다. 블록의 배치는 실제 열 인덱스와 무관한 모식도입니다.';
      $('tensor-path').textContent='layers.{15,19}.mlp.down_proj.weight';
      return;
    }
    const edit=edits.get(selected),type=metadata.attentionTypes[selected]==='full_attention'?'Full attention':'Linear attention';
    $('inspector-title').textContent=`Layer ${String(selected).padStart(2,'0')}${edit?' · 수정됨':''}`;
    $('inspector-title').style.color=selected===15?'#b3a0ff':selected===19?'#a4f0d2':'#d8e5f5';
    $('inspector-summary').textContent=edit?`${selected+1}번째 레이어입니다. Attention은 그대로 두고 MLP의 down_proj 가중치 중 64개 열을 수정했습니다.`:`${selected+1}번째 레이어입니다. 이번 실험에서는 이 레이어의 가중치를 수정하지 않았습니다.`;
    $('fact-a-label').textContent='변경된 가중치 값';$('fact-a').textContent=(edit?.changedValues??0).toLocaleString('ko-KR');
    $('fact-b-label').textContent='Attention 유형';$('fact-b').textContent=type;
    $('fact-c-label').textContent='down_proj 행렬';$('fact-c').textContent='1,024 × 3,584';
    $('inspector-note').textContent=edit?'64개의 발광 블록은 수정한 열을 나타냅니다. 실제로 값이 달라진 개수는 BF16 반올림 후 집계했습니다.':'회색 모듈은 원본 가중치를 유지합니다. 구조상의 색·크기·밝기는 모델 활성값을 표현하지 않습니다.';
    $('tensor-path').textContent=`model.language_model.layers.${selected}.mlp.down_proj.weight`;
  }
  function applyVisibility(){
    layers.forEach((obj,index)=>{obj.visible=!isolate||edits.has(index);});
    $('view-isolate').setAttribute('aria-pressed',String(isolate));
    $('view-mode').textContent=isolate?'EDITED LAYERS / 15 + 19':'24 LAYERS / 2 EDITED';dirty=true;
  }
  function resize(){
    const width=viewport.clientWidth,height=viewport.clientHeight;
    const aspect=width/height,halfHeight=Math.max(fitHalfHeight,fitHalfWidth/aspect);
    renderer.setSize(width,height);
    camera.left=-halfHeight*aspect;camera.right=halfHeight*aspect;
    camera.top=halfHeight;camera.bottom=-halfHeight;
    camera.updateProjectionMatrix();dirty=true;
  }
  new ResizeObserver(resize).observe(viewport);resize();overview();
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;dirty=true;}).observe(viewport);
  for(const b of buttons){b.disabled=false;b.addEventListener('click',()=>focus(Number(b.dataset.layer)));}
  document.querySelectorAll('[data-focus]').forEach(b=>{b.disabled=false;b.addEventListener('click',()=>focus(Number(b.dataset.focus)));});
  for(const id of ['view-reset','view-isolate','view-rotate','layer-spacing'])$(id).disabled=false;
  $('view-reset').addEventListener('click',()=>{
    spacing=1;$('layer-spacing').value='1';$('spacing-value').textContent='1.0×';
    setSpacing();isolate=false;applyVisibility();overview();
  });
  $('view-isolate').addEventListener('click',()=>{isolate=!isolate;applyVisibility();if(isolate&&selected!==null&&!edits.has(selected)){selected=null;updateInspector();}});
  $('view-rotate').addEventListener('click',()=>{controls.autoRotate=!controls.autoRotate;$('view-rotate').setAttribute('aria-pressed',String(controls.autoRotate));dirty=true;});
  function setSpacing(){
    layers.forEach((obj,i)=>{obj.position.copy(originalPositions.get(obj));obj.position.x+=i*LAYER_GAP*(spacing-1);});
    model.traverse(obj=>{
      if(obj.userData.stage==='output'){obj.position.copy(originalPositions.get(obj));obj.position.x+=(spacing-1)*23*LAYER_GAP;}
    });
    model.updateMatrixWorld(true);
  }
  $('layer-spacing').addEventListener('input',e=>{
    spacing=Number(e.target.value);$('spacing-value').textContent=spacing.toFixed(1)+'×';
    setSpacing();
    if(selected===null)overview();else focus(selected);
    dirty=true;
  });
  const raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2();
  const activePointers=new Map();let gestureMoved=false,multiTouch=false,lastZoom=-Infinity;
  function hitLayer(event){
    const rect=canvas.getBoundingClientRect();
    pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);
    raycaster.setFromCamera(pointer,camera);
    const objects=[...layers.values()].filter(o=>o.visible);
    for(const hit of raycaster.intersectObjects(objects,true)){
      for(let obj=hit.object;obj;obj=obj.parent)if(Number.isInteger(obj.userData.layerIndex))return obj.userData.layerIndex;
    }
    return null;
  }
  canvas.addEventListener('wheel',()=>{lastZoom=performance.now();$('layer-tooltip').hidden=true;},{passive:true});
  canvas.addEventListener('pointerdown',event=>{
    if(!activePointers.size){gestureMoved=false;multiTouch=false;}
    activePointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
    if(activePointers.size>1)multiTouch=true;
    $('layer-tooltip').hidden=true;
  });
  canvas.addEventListener('pointerup',event=>{
    const start=activePointers.get(event.pointerId);
    const click=start&&event.button===0&&!multiTouch&&!gestureMoved&&activePointers.size===1
      &&performance.now()-lastZoom>250&&Math.hypot(event.clientX-start.x,event.clientY-start.y)<5;
    activePointers.delete(event.pointerId);
    if(click){const index=hitLayer(event);if(index!==null)focus(index);}
  });
  canvas.addEventListener('pointercancel',event=>{activePointers.delete(event.pointerId);gestureMoved=true;});
  canvas.addEventListener('pointermove',event=>{
    const start=activePointers.get(event.pointerId);
    if(start&&Math.hypot(event.clientX-start.x,event.clientY-start.y)>=5)gestureMoved=true;
    if(event.buttons||event.pointerType==='touch')return;
    const index=hitLayer(event),tooltip=$('layer-tooltip');tooltip.hidden=index===null;
    if(index!==null){const rect=canvas.getBoundingClientRect();tooltip.textContent=`LAYER ${index}${edits.has(index)?' / MLP EDITED':''}`;tooltip.style.left=Math.min(event.clientX-rect.left+14,rect.width-180)+'px';tooltip.style.top=Math.max(30,event.clientY-rect.top-32)+'px';}
  });
  canvas.addEventListener('pointerleave',()=>{$('layer-tooltip').hidden=true;});
  canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();$('model-loading').hidden=false;$('model-loading').textContent='3D 연결이 중단되었습니다. 페이지를 새로고침해 주세요.';});
  $('model-loading').hidden=true;
  viewport.dataset.ready='true';
  renderer.setAnimationLoop(()=>{
    if(!visible||document.hidden)return;
    if(controls.update()||dirty||controls.autoRotate){renderer.render(scene,camera);dirty=false;}
  });
}

init().catch(error=>{
  console.error('3D viewer:',error);
  viewport.querySelector('canvas')?.remove();
  $('model-loading').hidden=false;$('model-loading').textContent='3D 화면을 만들지 못했습니다. WebGL을 지원하는 브라우저에서 다시 열어 주세요.';
  viewport.dataset.ready='error';
});
