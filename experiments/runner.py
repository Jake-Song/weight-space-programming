"""Trusted host controller. Editor code never runs in this process."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
EDITOR = 'wsp-editor:local'
GPU = 'wsp-gpu:local'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def mount(source, target, readonly=True):
    source = Path(source).resolve()
    if ',' in str(source):
        raise ValueError('Docker mount paths cannot contain commas')
    return ['--mount', f'type=bind,source={source},target={target}' + (',readonly' if readonly else '')]


def isolated():
    return ['docker', 'run', '--rm', '--network', 'none', '--read-only',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges=true',
            '--pids-limit', '512', '--user', f'{os.getuid()}:{os.getgid()}',
            '--tmpfs', '/tmp:rw,nosuid,size=8g', '--shm-size', '4g']


@contextmanager
def locked(root):
    with (root / '.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def validate_request(req, practice):
    if req.get('type') not in ('generate', 'teacher_forced'):
        raise ValueError('type must be generate or teacher_forced')
    model = req.get('model', 'edited')
    if model not in ('original', 'edited', 'both'):
        raise ValueError('Invalid model')
    if req['type'] == 'generate':
        ids = req.get('problems', [])
        prompts = req.get('prompts', [])
        n = req.get('n', 1)
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError('n must be a positive integer')
        if not isinstance(ids, list) or not isinstance(prompts, list) or not all(isinstance(x, str) for x in ids + prompts):
            raise ValueError('problems/prompts must be string lists')
        if len(set(ids)) != len(ids) or any(x not in practice for x in ids):
            raise ValueError('Duplicate or unknown practice id')
        if not 1 <= (len(ids) + len(prompts)) * n * (2 if model == 'both' else 1) <= 512:
            raise ValueError('Generation budget must be 1..512, across both models')
        if not isinstance(req.get('max_tokens', 32768), int) or not 1 <= req.get('max_tokens', 32768) <= 32768:
            raise ValueError('max_tokens must be 1..32768')
        if req.get('activations'):
            raise ValueError('Use teacher_forced for activations; generation hooks are not implemented')
    else:
        if model == 'both':
            raise ValueError('teacher_forced accepts one model')
        texts = req.get('texts', [])
        if not isinstance(texts, list) or not 1 <= len(texts) <= 1024 or not all(isinstance(x, str) and x for x in texts):
            raise ValueError('texts must contain 1..1024 nonempty strings')
        ret = req.get('return', {})
        if set(ret) - {'token_logprobs', 'hidden_states', 'mlp_act'}:
            raise ValueError('Supported returns: token_logprobs, hidden_states, mlp_act')
        for key in ('hidden_states', 'mlp_act'):
            layers = ret.get(key, [])
            upper = 24 if key == 'hidden_states' else 23
            if not isinstance(layers, list) or any(type(x) is not int or not 0 <= x <= upper for x in layers):
                raise ValueError('Invalid layer indices')
    allowed = {'type', 'model', 'problems', 'prompts', 'n', 'max_tokens'} if req['type'] == 'generate' else {'type', 'model', 'texts', 'return'}
    if set(req) - allowed:
        raise ValueError(f'Unsupported request fields: {set(req) - allowed}')
    return req


def editor_start(root, state):
    cmd = isolated()
    cmd.remove('--rm')
    cmd += ['--detach', '--name', state['container'], '--memory', '96g', '--cpus', '2']
    cmd += mount(root / 'assets/model', '/model') + mount(root / 'assets/practice', '/practice')
    cmd += mount(root / 'work', '/work', False) + [state['editor_image']]
    subprocess.run(cmd, check=True)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        ready = subprocess.run(['docker', 'exec', state['container'], 'test', '-S', '/tmp/kernel.sock'],
                               capture_output=True)
        if ready.returncode == 0:
            return
        time.sleep(.25)
    raise RuntimeError('Kernel not ready after 60 seconds; inspect docker logs and restart if needed')


def stop_editor(state):
    names = subprocess.check_output(['docker', 'ps', '-a', '--filter',
                'name=^/' + state['container'] + '$', '--format', '{{.ID}}'], text=True).strip()
    if names:
        subprocess.run(['docker', 'rm', '-f', state['container']], check=True)


def snapshot(root, destination):
    destination.mkdir()
    hashes = {}
    for p in (root / 'work/edited').glob('*'):
        if p.is_symlink() or not p.is_file() or not re.fullmatch(r'model\.language_model\.layers\.\d+\.[A-Za-z0-9_.]+\.npy', p.name):
            raise ValueError(f'Invalid edit artifact: {p.name}')
        dst = destination / p.name
        shutil.copyfile(p, dst)
        hashes[p.name] = sha(dst)
    return hashes


def gpu_job(root, state, job, online=False):
    cmd = isolated()
    if online:
        cmd[cmd.index('none')] = 'bridge'
    else:
        cmd += ['--gpus', 'all', '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1']
    cmd += mount(root / 'assets', '/assets', not online) + mount(job, '/job', False)
    cmd += [state['gpu_image'], 'prepare' if online else 'run']
    image_id = subprocess.check_output(['docker', 'image', 'inspect', state['gpu_image'], '--format', '{{.Id}}'], text=True).strip()
    write(job / 'runtime.json', {'image_id': image_id, 'command': cmd, 'started': time.time()})
    with (job / 'worker.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=REPO / '.sandbox')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('build')
    init = sub.add_parser('init')
    init.add_argument('--budget', type=int, choices=[10, 25, 50], default=50)
    for name in ('prepare', 'start', 'stop', 'status', 'freeze', 'evaluate'):
        sub.add_parser(name)
    code = sub.add_parser('python')
    code.add_argument('file', type=Path)
    code.add_argument('--explain', required=True)
    forward = sub.add_parser('forward')
    forward.add_argument('request', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    if a.command == 'build':
        for role, tag in [('editor', EDITOR), ('gpu', GPU)]:
            subprocess.run(['docker', 'build', '-f', str(REPO / f'experiments/Dockerfile.{role}'),
                            '-t', tag, str(REPO)], check=True)
        return
    if a.command == 'init':
        root.mkdir(parents=True, exist_ok=True)
        with locked(root):
            if (root / 'state.json').exists():
                raise ValueError('Run already exists; choose a new --root')
            for directory in ('assets', 'work/edited', 'jobs', 'code'):
                (root / directory).mkdir(parents=True, exist_ok=True)
            write(root / 'state.json', dict(budget=a.budget, used=0, python_runs=0,
                  frozen=False, container='wsp-' + uuid.uuid4().hex[:12],
                  editor_image=EDITOR, gpu_image=GPU))
        return
    with locked(root):
        state = read(root / 'state.json')
        # Recover successful passes even if the controller exited before updating state.
        state['used'] = sum((job / 'output/provenance.json').exists()
                            for job in (root / 'jobs').glob('forward-*'))
        write(root / 'state.json', state)
        if a.command == 'status':
            print(json.dumps(state, indent=2))
            return
        if a.command == 'prepare':
            if (root / 'assets/manifest.json').exists():
                raise ValueError('Assets already prepared')
            job = root / 'jobs' / ('prepare-' + uuid.uuid4().hex[:8])
            job.mkdir()
            gpu_job(root, state, job, online=True)
        elif a.command == 'start':
            if state['frozen'] or not (root / 'assets/manifest.json').exists():
                raise ValueError('Run frozen or assets not prepared')
            editor_start(root, state)
        elif a.command == 'stop':
            stop_editor(state)
        elif a.command == 'python':
            if state['frozen'] or state['python_runs'] >= 300:
                raise ValueError('Frozen or 300 Python executions exhausted')
            code = a.file.read_text()
            if not a.explain.strip():
                raise ValueError('An explanation is required')
            state['python_runs'] += 1
            number = state['python_runs']
            write(root / 'state.json', state)
            (root / f'code/{number:03d}.py').write_text(code)
            (root / f'code/{number:03d}.txt').write_text(a.explain)
            try:
                result = subprocess.run(['docker', 'exec', '-i', state['container'], 'python', '/app/kernel.py', 'client'],
                                        input=code, text=True, capture_output=True, timeout=900, check=True)
            except subprocess.TimeoutExpired:
                stop_editor(state)
                raise RuntimeError('900-second limit: kernel stopped; restart loses variables, preserves edits')
            (root / f'code/{number:03d}.json').write_text(result.stdout)
            print(result.stdout)
        elif a.command == 'forward':
            if state['frozen'] or state['used'] >= state['budget']:
                raise ValueError('Frozen or forward budget exhausted')
            practice = {json.loads(line)['id'] for path in (root / 'assets/practice').glob('*.jsonl')
                        for line in path.read_text().split("\n") if line.strip()}
            req = validate_request(read(a.request), practice)
            job = root / 'jobs' / ('forward-' + uuid.uuid4().hex[:12])
            job.mkdir()
            hashes = snapshot(root, job / 'edits')
            write(job / 'request.json', req)
            write(job / 'edit_hashes.json', hashes)
            # Only a successfully finished worker consumes a pass. Jobs never resume silently.
            gpu_job(root, state, job)
            state['used'] += 1
            write(root / 'state.json', state)
            dst = root / 'work' / job.name
            shutil.copytree(job / 'output', dst)
            print(dst)
        elif a.command == 'freeze':
            if state['frozen']:
                raise ValueError('Already frozen')
            if state['used'] != state['budget']:
                raise ValueError('Use all forward passes before final submission, as in the original')
            stop_editor(state)
            hashes = snapshot(root, root / 'submission')
            write(root / 'submission.json', {'hashes': hashes, 'created': time.time()})
            state['frozen'] = True
            write(root / 'state.json', state)
        elif a.command == 'evaluate':
            if not state['frozen']:
                raise ValueError('Freeze the final submission first')
            expected = read(root / 'submission.json')['hashes']
            actual = {p.name: sha(p) for p in (root / 'submission').iterdir()}
            if actual != expected:
                raise ValueError('Submission changed since freeze')
            for model in ('original', 'edited'):
                for bench, n in [('math500', 4), ('gsm8k', 2)]:
                    job = root / 'jobs' / f'eval-{model}-{bench}'
                    if (job / 'output/provenance.json').exists():
                        print(job / 'output/scores.json')
                        continue
                    if job.exists():
                        job.rename(job.with_name(job.name + '-failed-' + uuid.uuid4().hex[:8]))
                    job.mkdir()
                    shutil.copytree(root / 'submission', job / 'edits')
                    write(job / 'request.json', dict(type='generate', model=model, benchmark=bench,
                                                     n=n, max_tokens=32768))
                    gpu_job(root, state, job)
                    print(job / 'output/scores.json')


if __name__ == '__main__':
    main()
