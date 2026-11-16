"""One exploratory control: same edited columns, random rank-one direction.

Preserve each column's edit magnitude before BF16 rounding; report actual norms.
The column-selection information is intentionally shared with the published edit.
"""
import json, pathlib, shutil, torch
from safetensors.torch import load_file, save_file
root=pathlib.Path('/content/weight-space')
base=load_file(str(root/'models/base/model.safetensors'))
edited=load_file(str(root/'models/both/model.safetensors'))
out=root/'models/random'
out.mkdir(exist_ok=True)
g=torch.Generator().manual_seed(20261116)
audit=[]
for name in base:
    if torch.equal(base[name],edited[name]): continue
    w=base[name].float()
    delta=edited[name].float()-w
    u=torch.randn(w.shape[0],generator=g)
    u/=u.norm()
    # Random signs preserve shared rank-one structure and per-column magnitudes.
    signs=torch.randint(0,2,(w.shape[1],),generator=g)*2-1
    noise=u[:,None]*delta.norm(dim=0)[None,:]*signs[None,:]
    new=(w+noise).to(base[name].dtype)
    audit.append({'name':name,'published_norm':float(delta.norm()),
                  'random_actual_norm':float((new.float()-w).norm()),
                  'columns':int((delta!=0).any(dim=0).sum())})
    base[name]=new
save_file(base,str(out/'model.safetensors'),metadata={'format':'pt'})
for p in (root/'models/base').iterdir():
    if p.suffix!='.safetensors': shutil.copy2(p,out/p.name)
(root/'random_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit),flush=True)
