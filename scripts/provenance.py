import hashlib, importlib.metadata, json, pathlib, platform, subprocess, urllib.request
root=pathlib.Path('/content/weight-space')
with urllib.request.urlopen('https://api.github.com/repos/RikiyaT/lm_understand-logs/commits/main') as r:
    commit=json.load(r)['sha']
hashes={}
for rel in ['code/apply_edit.py','code/st_numpy.py','code/mathqa.py','runs/p50c/edit/manifest.json','runs/p50c/edit/edit.npz']:
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{commit}/exp8_math/{rel}') as r:
        data=r.read()
    local=root/'vendor'/rel
    if rel=='code/mathqa.py': local=local.with_suffix('.original.py')
    assert data==local.read_bytes(), f'Source changed: {rel}'
    hashes[rel]=hashlib.sha256(data).hexdigest()
result={'source_repository':'https://github.com/RikiyaT/lm_understand-logs',
        'source_head_observed':commit,'base_revision':'2fc06364715b967f1860aea9cf38778875588b17',
        'source_file_sha256':hashes,'python':platform.python_version(),
        'packages':{p:importlib.metadata.version(p) for p in ['torch','vllm','transformers','datasets','safetensors','math-verify']},
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True)}
(root/'provenance.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
