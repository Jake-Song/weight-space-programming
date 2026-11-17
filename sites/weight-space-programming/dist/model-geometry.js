import * as THREE from 'three';

export const LAYER_GAP=.6;

// All geometry is constructed in the browser from the experiment's metadata.
export function createModel(metadata){
  const model=new THREE.Group();model.name='Qwen_text_backbone';
  model.userData={generator:'JavaScript / Three.js',layout:'horizontal'};
  const edits=new Map(metadata.edits.map(edit=>[edit.layer,edit]));
  const candidates=new Map((metadata.candidates??[]).map(c=>[c.layer,c]));
  const highlights=new Map([...edits,...candidates]);
  const box=new THREE.BoxGeometry(1,1,1);
  const frame=new THREE.EdgesGeometry(box);
  const makeMaterial=(color,emission=0)=>new THREE.MeshStandardMaterial({
    color,roughness:.48,metalness:.25,emissive:color,emissiveIntensity:emission
  });
  const body=makeMaterial('#25354a'),attention=makeMaterial('#3c5a73'),mlp=makeMaterial('#31475e');
  const accent={},selectedBody={},glow={};
  const edgeMaterials={base:new THREE.LineBasicMaterial({color:'#506d89'})};
  for(const [layer,entry] of highlights){
    const color=entry.color??(layer===15?'#b3a0ff':'#a4f0d2');
    accent[layer]=color;selectedBody[layer]=makeMaterial(color,.08);glow[layer]=makeMaterial(color,.6);
    edgeMaterials[layer]=new THREE.LineBasicMaterial({color});
  }
  const addBox=(parent,name,position,size,material)=>{
    const mesh=new THREE.Mesh(box,material);mesh.name=name;
    mesh.position.set(...position);mesh.scale.set(...size);parent.add(mesh);return mesh;
  };
  function label(parent,text,position,width,height,color){
    const canvas=document.createElement('canvas');
    const ctx=canvas.getContext('2d');ctx.font='600 64px monospace';
    canvas.width=Math.ceil(ctx.measureText(text).width)+24;canvas.height=96;
    ctx.font='600 64px monospace';
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle=color;
    ctx.fillText(text,canvas.width/2,48);
    const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
    const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,transparent:true,depthTest:false,toneMapped:false}));
    sprite.name=`Label_${text}`;sprite.userData.label=text;
    sprite.position.set(...position);sprite.scale.set(width,height,1);sprite.renderOrder=5;
    parent.add(sprite);
  }
  const start=-(metadata.layers-1)*LAYER_GAP/2;
  for(let index=0;index<metadata.layers;index++){
    const group=new THREE.Group(),edit=edits.get(index),candidate=candidates.get(index),highlight=highlights.get(index);
    group.name=`Layer_${String(index).padStart(2,'0')}`;group.position.x=start+index*LAYER_GAP;
    group.userData={layerIndex:index,edited:!!edit,candidate:!!candidate,attentionType:metadata.attentionTypes[index],changedValues:edit?.changedValues??0};
    model.add(group);
    addBox(group,'Layer chassis',[0,0,0],[.09,4,2.8],highlight?selectedBody[index]:body);
    const outline=new THREE.LineSegments(frame,edgeMaterials[highlight?index:'base']);
    outline.scale.set(.11,4.02,2.82);group.add(outline);
    addBox(group,'Attention',[.11,1,0],[.12,1.65,2.45],attention).userData.component='attention';
    addBox(group,'MLP',[.11,-1,0],[.12,1.65,2.45],mlp).userData.component='mlp';
    if(highlight){
      // Marks represent candidate channels or the prior edit columns; placement is schematic.
      const count=candidate?candidate.channels.length:edit.columns;
      const columns=new THREE.InstancedMesh(box,glow[index],count);
      columns.name=`Layer_${index}_edited_columns`;columns.userData.component=candidate?'candidate_channels':'edited_columns';
      const transform=new THREE.Object3D();
      for(let i=0;i<count;i++){
        transform.position.set(.23,candidate?-1:-1.66+Math.floor(i/8)*.19,candidate?(i-(count-1)/2)*.4:-1.04+(i%8)*.295);
        transform.scale.set(.09,candidate ? .3 : .115,candidate ? .25 : .18);transform.updateMatrix();columns.setMatrixAt(i,transform.matrix);
      }
      columns.instanceMatrix.needsUpdate=true;columns.computeBoundingBox();columns.computeBoundingSphere();group.add(columns);
      addBox(group,'Highlight rail',[.04,-2.1,0],[.16,.12,2.9],glow[index]);
    }
    label(group,String(index).padStart(2,'0'),[0,-2.65,1.35],highlight ? .62 : .43,highlight ? .44 : .3,accent[index]??'#a6bdd4');
  }
  for(const [stage,x,text] of [['embedding',start-1,'INPUT'],['output',-start+1,'OUTPUT']]){
    const group=new THREE.Group();group.name=stage;group.position.x=x;group.userData.stage=stage;model.add(group);
    addBox(group,stage,[0,0,0],[.25,4.4,3.1],body);
    const outline=new THREE.LineSegments(frame,edgeMaterials.base);outline.scale.set(.26,4.41,3.11);group.add(outline);
    label(group,text,[0,2.85,0],1.35,.5,'#cad9eb');
  }
  model.updateMatrixWorld(true);
  return model;
}
