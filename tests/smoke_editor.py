"""Actual Docker integration check, using tiny synthetic tensors and no GPU/downloads.
Run after building wsp-editor:local: python3 tests/smoke_editor.py
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import runner as R


def main():
    with tempfile.TemporaryDirectory(prefix='wsp-smoke-') as directory:
        root = Path(directory)
        for name in ('assets/model', 'assets/practice', 'work/edited'):
            (root / name).mkdir(parents=True)
        code = '''
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models
save_file({'model.language_model.layers.15.mlp.down_proj.weight': torch.tensor([[1.,2.],[3.,4.]]),
           'model.language_model.norm.weight': torch.ones(2)}, '/fixture/model.safetensors')
Tokenizer(models.WordLevel({'[UNK]':0}, unk_token='[UNK]')).save('/fixture/tokenizer.json')
'''
        subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--user', f'{R.os.getuid()}:{R.os.getgid()}',
                        *R.mount(root / 'assets/model', '/fixture', False), '--entrypoint', 'python',
                        R.EDITOR, '-c', code], check=True)
        state = {'container': 'wsp-smoke-' + R.uuid.uuid4().hex[:12], 'editor_image': R.EDITOR}
        R.editor_start(root, state)
        try:
            for _ in range(60):
                check = subprocess.run(['docker', 'exec', state['container'], 'test', '-S', '/tmp/kernel.sock'],
                                       capture_output=True)
                if check.returncode == 0:
                    break
                time.sleep(.5)
            else:
                raise RuntimeError('Kernel did not become ready')

            def execute(code):
                result = subprocess.run(['docker', 'exec', '-i', state['container'], 'python', '/app/kernel.py', 'client'],
                                        input=code, text=True, capture_output=True, check=True, timeout=30)
                return json.loads(result.stdout)

            assert execute('answer = 41')['ok']
            assert execute('answer + 1')['output'].strip() == '42'
            assert execute("print(torch.cuda.is_available(), torch.is_grad_enabled())")['output'].strip() == 'False False'
            assert not execute("from pathlib import Path\nPath('/model/forbidden').write_text('no')")['ok']
            assert not execute("Path('/forbidden').write_text('no')")['ok']
            assert not execute("Path('/assets/bench/math500.jsonl').read_text()")['ok']
            assert not execute("save_edit('model.language_model.norm.weight', torch.ones(2))")['ok']
            tensor = 'model.language_model.layers.15.mlp.down_proj.weight'
            assert not execute(f"save_edit('{tensor}', load_original('{tensor}') * 2)")['ok']
            assert execute(f"w = load_original('{tensor}')\nw[0,0] += 1\nsave_edit('{tensor}', w)")['ok']
            assert execute('edited()')['output'].find(tensor) >= 0
            assert execute("len(Path('/proc/net/route').read_text().splitlines())")['output'].strip() == '1'
            cfg = json.loads(subprocess.check_output(['docker', 'inspect', state['container']], text=True))[0]
            assert cfg['HostConfig']['NetworkMode'] == 'none'
            assert cfg['HostConfig']['ReadonlyRootfs']
            assert not cfg['HostConfig']['DeviceRequests']
            # The trusted worker must reject files written around save_edit too.
            (root / 'job/edits').mkdir(parents=True)
            (root / 'job/output').mkdir()
            validation = '''
import sys
sys.path.insert(0, '/app')
import numpy as np
import worker
name = 'model.language_model.layers.15.mlp.down_proj.weight'
p = '/job/edits/' + name + '.npy'
original = np.array([[1.,2.],[3.,4.]], dtype=np.float32)
for bad in (original * 2, np.full((2,2), np.nan, dtype=np.float32), np.ones(3, dtype=np.float32)):
    np.save(p, bad)
    try:
        worker.checkpoint('edited')
    except ValueError:
        pass
    else:
        raise AssertionError('Untrusted edit accepted')
original[0,0] += 1
np.save(p, original)
worker.checkpoint('edited')
print('PASS: independent checkpoint validator rejects scaling, NaN and shape mismatch')
'''
            subprocess.run(['docker', 'run', '--rm', '--network', 'none',
                            '--user', f'{R.os.getuid()}:{R.os.getgid()}',
                            *R.mount(root / 'assets', '/assets'), *R.mount(root / 'job', '/job', False),
                            '--entrypoint', 'python', R.EDITOR, '-c', validation], check=True)
            print('PASS: persistent kernel, CPU/no-grad, no network/GPU, read-only model/root, hidden benchmarks, edit restrictions')
        finally:
            subprocess.run(['docker', 'rm', '-f', state['container']], check=True, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
