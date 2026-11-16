import json, pathlib, torch
from safetensors import safe_open
root=pathlib.Path('/content/weight-space')
records=[]; directions=[]
with safe_open(str(root/'models/base/model.safetensors'),framework='pt') as b, safe_open(str(root/'models/both/model.safetensors'),framework='pt') as e:
    for l in [15,19]:
        name=f'model.language_model.layers.{l}.mlp.down_proj.weight'
        w=b.get_tensor(name).float(); delta=e.get_tensor(name).float()-w
        cols=(delta!=0).any(dim=0)
        u,s,v=torch.linalg.svd(delta[:,cols],full_matrices=False)
        directions.append(u[:,0])
        records.append({'layer':l,'changed_columns':int(cols.sum()),
            'delta_frobenius_norm':float(delta.norm()),
            'relative_frobenius_norm':float(delta.norm()/w.norm()),
            'leading_singular_energy_fraction':float(s[0]**2/(s**2).sum())})
result={'layers':records,'absolute_cosine_of_leading_edit_directions':float(torch.abs(directions[0]@directions[1]))}
(root/'weight_analysis.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
