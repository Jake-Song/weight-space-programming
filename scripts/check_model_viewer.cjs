// Run with Playwright installed, or set PLAYWRIGHT_MODULE to its absolute path.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const root=path.resolve(__dirname,'../sites/weight-space-programming/dist');
const close=(a,b,message)=>{
  assert.equal(a.length,b.length,message);
  a.forEach((v,i)=>assert(Math.abs(v-b[i])<1e-8,`${message}: ${v} != ${b[i]}`));
};
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_CHROMIUM||undefined,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1100},hasTouch:true});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    const requests=[];page.on('request',r=>requests.push(r.url()));
    // Inject a read-only camera probe into the test response, never production code.
    await page.route('**/model.js*',route=>{
      const source=fs.readFileSync(path.join(root,'model.js'),'utf8').replace("viewport.dataset.ready='true';",`
        window.readViewer=()=>({
          position:camera.position.toArray(),target:controls.target.toArray(),
          rotation:camera.quaternion.toArray(),zoom:camera.zoom,selected,spacing,isolate,
          layerCount:layers.size,view,
          candidateCounts:[10,19,23].map(i=>{let count=0;layers.get(i).traverse(o=>{if(o.userData.component==='candidate_channels')count+=o.count;});return count;}),
          columnCounts:[15,19].map(i=>{let count=0;layers.get(i).traverse(o=>{if(o.userData.component==='edited_columns')count+=o.count;});return count;}),
          endpoints:[0,23].map(i=>layers.get(i).getWorldPosition(new THREE.Vector3()).toArray()),
          layer19:layers.get(19).getWorldPosition(new THREE.Vector3()).project(camera).toArray()
        });
        viewport.dataset.ready='true';`);
      return route.fulfill({contentType:'text/javascript',body:source});
    });
    await page.goto('http://127.0.0.1:4173/model.html',{waitUntil:'networkidle'});
    assert.equal(new URL(page.url()).pathname,'/','Old model URL forwards to the unified page');
    await page.waitForFunction(()=>document.querySelector('#viewport').dataset.ready==='true');
    assert.deepEqual(await page.locator('main > section').evaluateAll(nodes=>nodes.map(n=>n.id)),['diagnostic','model','run-history','results']);
    assert.equal(await page.locator('#question,#methods,#takeaway,.model-notes,.sources,.hero').count(),0);
    const state=()=>page.evaluate(()=>window.readViewer());
    await page.waitForFunction(()=>document.querySelector('#diagnostic').dataset.ready==='true');
    const data=JSON.parse(fs.readFileSync(path.join(root,'diagnostics.json'),'utf8'));
    const source=JSON.parse(fs.readFileSync(path.resolve(__dirname,'../results/runs/ten-steps-20261117-metrics.json'),'utf8'));
    for(const c of data.candidates){
      assert.deepEqual(c.ratios,source.by_layer[c.layer].ratios);
      assert.equal(c.criterion_met,c.ratios.every(r=>r>1)&&c.ratios.filter(r=>r>=1.25).length>=3);
    }
    assert.equal(await page.locator('.step-item').count(),10);
    assert.equal((await state()).view,'diagnostic');
    assert.deepEqual((await state()).candidateCounts,[3,3,5]);
    assert.deepEqual((await state()).columnCounts,[0,0]);
    assert.equal(await page.locator('#fact-a').innerText(),'0');
    for(const candidate of data.candidates){
      await page.locator(`[data-candidate="${candidate.layer}"]`).click();
      assert.deepEqual(await page.locator('.ratio-value').allTextContents(),candidate.ratios.map(r=>r.toFixed(4)+'×'));
      assert.equal(await page.locator('#candidate-verdict .verdict-badge').innerText(),candidate.criterion_met?'기준 충족':'기준 미충족');
      assert.equal((await state()).selected,candidate.layer);
      assert.equal(await page.locator('#fact-a').innerText(),'0');
    }
    await page.locator('[data-candidate="10"]').click();
    await page.locator('.diagnostic-table summary').click();
    assert.equal(await page.locator('#candidate-table tr').count(),3);
    for(const asset of ['reports/ten-steps-20261117.md','reports/ten-steps-20261117-metrics.json'])assert((await page.request.get('http://127.0.0.1:4173/'+asset)).ok());
    await page.locator('.step-item').nth(1).locator('summary').click();
    assert.match(await page.locator('.step-item').nth(1).innerText(),/토크나이저/);
    assert.equal(await page.locator('.step-item').nth(1).locator('.step-title').evaluate(e=>getComputedStyle(e).transform),'none');
    await page.locator('#view-reset').click();
    await page.evaluate(()=>scrollTo(0,0));
    await page.screenshot({path:'/tmp/weight-space-diagnostic-desktop.png'});
    await page.locator('[data-model-mode="published"]').click();
    const initial=await state();
    assert.equal(initial.layerCount,24);assert.deepEqual(initial.columnCounts,[64,64]);
    assert(initial.endpoints[1][0]-initial.endpoints[0][0]>10,'Layer progression is horizontal');
    assert.equal(initial.endpoints[1][1],initial.endpoints[0][1]);
    await page.locator('#viewport canvas').scrollIntoViewIfNeeded();
    const box=await page.locator('#viewport canvas').boundingBox();
    await page.mouse.move(box.x+box.width*.43,box.y+box.height*.45);
    for(let n=0;n<3;n++){
      await page.mouse.wheel(0,600);
      await page.waitForFunction(z=>window.readViewer().zoom<z,initial.zoom);
      const out=await state();close(out.target,initial.target,'Zoom-out keeps center');
      close(out.position,initial.position,'Orthographic zoom keeps camera position');
      await page.mouse.wheel(0,-600);
      await page.waitForFunction(z=>Math.abs(window.readViewer().zoom-z)<1e-8,initial.zoom);
      const back=await state();close(back.position,initial.position,'Zoom round trip position');
      close(back.target,initial.target,'Zoom round trip center');close(back.rotation,initial.rotation,'Zoom round trip rotation');
    }
    // An asymmetric pinch with one stationary finger must not select/recenter a layer.
    const cdp=await page.context().newCDPSession(page);
    const p=(await state()).layer19;
    const x=box.x+(p[0]+1)*box.width/2,y=box.y+(1-p[1])*box.height/2;
    const touch=(id,x,y)=>({id,x,y,radiusX:3,radiusY:3,force:1});
    await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[touch(1,x-80,y),touch(2,x,y)]});
    await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[touch(1,x-120,y),touch(2,x,y)]});
    await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[touch(1,x-120,y)]});
    await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
    const pinched=await state();assert.equal(pinched.selected,null,'Pinch does not select a layer');
    close(pinched.target,initial.target,'Pinch keeps center');
    await page.locator('#view-reset').click();
    await page.locator('[data-focus="15"]').click();
    const focus=await state();assert.equal(focus.selected,15);
    assert.equal(await page.locator('#fact-a').innerText(),'65,260');
    await page.mouse.move(box.x+box.width*.45,box.y+box.height*.4);
    await page.mouse.wheel(0,500);
    await page.waitForFunction(()=>window.readViewer().zoom<1);
    await page.mouse.wheel(0,-500);
    await page.waitForFunction(()=>Math.abs(window.readViewer().zoom-1)<1e-8);
    close((await state()).target,focus.target,'Selected-layer zoom keeps center');
    await page.locator('#view-isolate').click();
    await page.locator('#layer-spacing').fill('1.8');
    await page.locator('#view-rotate').click();
    await page.locator('#view-reset').click();
    const reset=await state();
    assert.equal(reset.spacing,1);assert.equal(reset.selected,null);assert.equal(reset.isolate,false);
    close(reset.position,initial.position,'Reset restores exact camera');
    close(reset.target,initial.target,'Reset restores exact center');
    close(reset.rotation,initial.rotation,'Reset restores exact orientation');
    assert.equal(await page.locator('#view-rotate').getAttribute('aria-pressed'),'false');
    assert.equal(await page.locator('#zoom-readout').innerText(),'100%');
    await page.screenshot({path:'/tmp/weight-space-horizontal-desktop.png',fullPage:true});
    await page.locator('[data-focus="19"]').click();
    await page.locator('#view-isolate').click();
    await page.screenshot({path:'/tmp/weight-space-horizontal-detail.png',fullPage:true});
    await page.locator('[data-model-mode="diagnostic"]').click();
    assert.deepEqual((await state()).candidateCounts,[3,3,5]);
    for(const width of [390,768]){
      await page.setViewportSize({width,height:900});await page.locator('#view-reset').click();
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      await page.locator('[data-candidate="19"]').click();
      assert.equal(await page.locator('#candidate-verdict .verdict-badge').innerText(),'기준 미충족');
      if(width===390){await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:'/tmp/weight-space-diagnostic-mobile.png',fullPage:true});}
    }
    assert.deepEqual(errors,[]);
    await page.goto('http://127.0.0.1:4173/',{waitUntil:'networkidle'});
    await page.waitForFunction(()=>document.querySelector('#viewport').dataset.ready==='true');
    assert.equal(await page.locator('.bar-row').count(),6);
    assert.equal(await page.locator('#delta-accuracy').innerText(),'+6%p');
    await page.locator('[data-metric="tokens"]').click();
    assert.equal(await page.locator('[data-condition="nothink"] .bar-value').innerText(),'8,245');
    await page.locator('[data-condition="nothink"]').click();
    assert.equal(await page.locator('#detail-title').innerText(),'원본 · thinking off');
    await page.locator('#from-condition').selectOption('base');
    assert.equal(await page.locator('#delta-accuracy').innerText(),'+49%p');
    await page.locator('.data-table summary').click();
    assert.equal(await page.locator('#table-body tr').count(),6);
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    assert(!requests.some(url=>/\.(glb|blend|png)(?:\?|$)/.test(url)),'No baked model or Blender image requests');
    assert(!requests.some(url=>url.includes('GLTFLoader')),'No model loader dependency');
    assert.deepEqual(errors,[]);
    console.log('PASS: current diagnostic ratios and verdicts, 10-step log, candidate markers, published edit toggle, historical charts, zoom/pinch/reset, desktop/mobile, no JS errors.');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
