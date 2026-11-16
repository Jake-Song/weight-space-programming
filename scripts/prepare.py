"""Download pinned base and published edits; build independently verified ablations."""
import hashlib, json, os, pathlib, random, shutil, subprocess, sys, urllib.request

ROOT = pathlib.Path('/content/weight-space')
ROOT.mkdir(exist_ok=True)
os.chdir(ROOT)
URL = 'https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/'
for rel in ['code/apply_edit.py', 'code/st_numpy.py', 'code/mathqa.py',
            'runs/p50c/edit/manifest.json', 'runs/p50c/edit/edit.npz']:
    dst = ROOT / 'vendor' / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL + rel, dst)

# The release omits mcqa.py. Its loop detector is unused by build_prompt,
# extract and is_correct. Remove only this import block, preserving grading.
qpath = ROOT/'vendor/code/mathqa.py'
source = qpath.read_text()
start = source.index('try:   # the multiple-choice')
end = source.index('\nREPO =', start)
qpath.with_suffix('.original.py').write_text(source)
qpath.write_text(source[:start] + source[end:])

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from datasets import load_dataset
import torch

revision = '2fc06364715b967f1860aea9cf38778875588b17'
base = pathlib.Path(snapshot_download('Qwen/Qwen3.5-0.8B', revision=revision,
    allow_patterns=['*.json', '*.jinja', '*.safetensors', 'tokenizer*', 'vocab*', 'merges*']))
edit_dir = ROOT / 'vendor/runs/p50c/edit'
original = {}
for shard in sorted(base.glob('*.safetensors')):
    original.update(load_file(str(shard)))
assert original, f'No weight tensors downloaded from {base}'
base_weights = ROOT/'base_merged.safetensors'
save_file(original, str(base_weights), metadata={'format':'pt'})
subprocess.run([sys.executable, str(ROOT/'vendor/code/apply_edit.py'),
    str(base_weights), str(edit_dir), str(ROOT/'edited_tensors')], check=True)
manifest = json.loads((edit_dir/'manifest.json').read_text())
import numpy as np
edited = {t['name']: torch.from_numpy(np.load(ROOT/'edited_tensors'/ (t['name']+'.npy'))).to(torch.bfloat16)
          for t in manifest['tensors']}
audit = []
for condition, layers in [('base', []), ('layer15', [15]), ('layer19', [19]), ('both', [15,19])]:
    out = ROOT/'models'/condition
    out.mkdir(parents=True, exist_ok=True)
    for src in base.iterdir():
        if src.is_file() and src.suffix != '.safetensors' and not src.name.endswith('.index.json'):
            shutil.copy2(src, out/src.name)
    state = dict(original)
    changed = []
    for name, tensor in edited.items():
        if any(f'.layers.{l}.' in name for l in layers):
            state[name] = tensor
            changed.append({'name': name, 'entries': int((tensor != original[name]).sum()),
                            'sha256': hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()})
    save_file(state, str(out/'model.safetensors'), metadata={'format':'pt'})
    audit.append({'condition':condition, 'changed':changed})
    print('BUILT', condition, changed, flush=True)
(ROOT/'weight_audit.json').write_text(json.dumps(audit, indent=2))
ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
indices = list(range(len(ds)))
random.Random(20261116).shuffle(indices)
for name, chosen in [('smoke',indices[:20]), ('heldout',indices[20:120])]:
    rows = [{k:ds[i][k] for k in ['unique_id','problem','answer','subject','level']} for i in chosen]
    (ROOT/f'{name}.json').write_text(json.dumps(rows, indent=2))
print('PREPARE_COMPLETE', flush=True)
