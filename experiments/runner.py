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

try:
    from . import continuity as C
except ImportError:
    import continuity as C

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
    inspect = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', state['container']],
                             text=True, capture_output=True)
    if inspect.returncode == 0:
        if inspect.stdout.strip() == 'true':
            ready = subprocess.run(['docker', 'exec', state['container'], 'test', '-S', '/tmp/kernel.sock'], capture_output=True)
            if ready.returncode == 0:
                return
            raise RuntimeError('Existing kernel is not ready; inspect logs or stop and restart it')
        stop_editor(state)
    cmd = isolated()
    cmd.remove('--rm')
    cmd += ['--detach', '--name', state['container'], '--memory', state.get('editor_memory', '96g'), '--cpus', '2']
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
    if (root / 'work/edited').is_symlink():
        raise ValueError('Edit directory cannot be a symlink')
    destination.mkdir()
    hashes = {}
    for p in (root / 'work/edited').glob('*'):
        if p.is_symlink() or not p.is_file() or not re.fullmatch(r'model\.language_model\.layers\.\d+\.[A-Za-z0-9_.]+\.npy', p.name):
            raise ValueError(f'Invalid edit artifact: {p.name}')
        dst = destination / p.name
        shutil.copyfile(p, dst)
        hashes[p.name] = sha(dst)
    return hashes


def freeze(root, state):
    """Freeze once; recover a submission persisted before a controller crash."""
    if state['used'] != state['budget']:
        raise ValueError('Use all forward passes before final submission, as in the original')
    stop_editor(state)
    manifest = root / 'submission.json'
    destination = root / 'submission'
    if manifest.exists():
        expected = read(manifest)['hashes']
        actual = {p.name: sha(p) for p in destination.iterdir()}
        if actual != expected:
            raise ValueError('Submission changed since freeze')
    else:
        stage = root / ('submission-' + uuid.uuid4().hex)
        hashes = snapshot(root, stage)
        if destination.exists():
            # A crash after rename but before manifest: only accept the same edits.
            if {p.name: sha(p) for p in destination.iterdir()} != hashes:
                raise ValueError('Uncommitted submission differs from current edits')
            shutil.rmtree(stage)
        else:
            stage.rename(destination)
        write(manifest, {'hashes': hashes, 'created': time.time()})
    state['frozen'] = True
    write(root / 'state.json', state)


def gpu_job(root, state, job, online=False):
    if not online and C.seal(job):
        return
    if state.get('colab_session'):
        try:
            from .colab_backend import gpu_job as colab_job
        except ImportError:
            from colab_backend import gpu_job as colab_job
        return colab_job(root, state, job, online)
    receipt = job / 'container.json'
    name = 'wsp-job-' + hashlib.sha256(str(job.resolve()).encode()).hexdigest()[:20]
    if receipt.exists():
        name = read(receipt)['name']
        running = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', name],
                                 text=True, capture_output=True)
        if running.returncode == 0 and running.stdout.strip() == 'true':
            subprocess.run(['docker', 'wait', name], capture_output=True)
            if not online and C.seal(job):
                return
            raise RuntimeError('Previous GPU container finished without a complete result; retry the same job ID')
        if running.returncode == 0:
            subprocess.run(['docker', 'rm', name], check=True, capture_output=True)
    C.write(receipt, {'name': name})
    cmd = isolated() + ['--name', name]
    if online:
        cmd[cmd.index('none')] = 'bridge'
    else:
        cmd += ['--gpus', 'all', '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1']
    cmd += mount(root / 'assets', '/assets', not online) + mount(job, '/job', False)
    cmd += [state['gpu_image'], 'prepare' if online else 'run']
    image_id = subprocess.check_output(['docker', 'image', 'inspect', state['gpu_image'], '--format', '{{.Id}}'], text=True).strip()
    write(job / 'runtime.json', {'image_id': image_id, 'command': cmd, 'started': time.time()})
    if (job / 'worker.log').exists():
        (job / 'worker.log').rename(job / ('worker.attempt-' + uuid.uuid4().hex[:8] + '.log'))
    with (job / 'worker.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def require_active(root, state):
    if state['frozen'] or C.ensure(root)['phase'] == 'paused':
        raise ValueError('Run frozen or paused; resume before changing the experiment')


def run_python(root, state, code, explanation, action_id=None):
    require_active(root, state)
    name = C.identifier(action_id or ('python-' + uuid.uuid4().hex[:12]))
    prefix = root / 'code' / name
    spec = dict(code_sha256=hashlib.sha256(code.encode()).hexdigest(), explanation=explanation)
    intent_path = prefix.with_name(prefix.name + '.intent.json')
    result_path = prefix.with_name(prefix.name + '.json')
    if intent_path.exists():
        intent = read(intent_path)
        if intent['spec'] != spec:
            raise ValueError('Python action ID already belongs to different code/explanation')
        if result_path.exists():
            return read(result_path)
    else:
        if state['python_runs'] >= state.get('python_budget', 300) or not explanation.strip():
            raise ValueError('Python budget exhausted or explanation missing')
        state['python_runs'] += 1
        write(root / 'state.json', state)
        C.write(intent_path, dict(id=name, spec=spec, status='pending'))
        prefix.with_name(prefix.name + '.py').write_text(code)
        prefix.with_name(prefix.name + '.txt').write_text(explanation)
        C.event(root, 'python_requested', action=name, explanation=explanation)
    envelope = json.dumps({'execution_id': name, 'code': code})
    try:
        process = subprocess.run(['docker', 'exec', '-i', state['container'], 'python', '/app/kernel.py', 'client'],
                                 input=envelope, text=True, capture_output=True, timeout=900, check=True)
    except subprocess.TimeoutExpired:
        stop_editor(state)
        raise RuntimeError('Kernel timed out. Resume the same action ID to inspect its receipt; do not blindly rerun code')
    result = json.loads(process.stdout)
    if result.get('uncertain'):
        raise RuntimeError('Python action was interrupted with an unknown outcome. Restore its pre-action checkpoint and abandon the agent step before continuing')
    if result.get('execution_id') != name or result.get('code_sha256') != spec['code_sha256']:
        raise RuntimeError('Kernel receipt protocol mismatch: rebuild the editor image, stop/restart the kernel, then retry the same action ID')
    C.write(result_path, result)
    C.write(intent_path, dict(id=name, spec=spec, status='complete'))
    C.event(root, 'python_finished', action=name, ok=result.get('ok'))
    return result


def run_forward(root, state, req, job_id=None, hypothesis_id=None):
    require_active(root, state)
    name = 'forward-' + C.identifier(job_id or uuid.uuid4().hex[:12])
    job = root / 'jobs' / name
    if job.exists():
        if read(job / 'request.json') != req:
            raise ValueError('Job ID already belongs to a different request')
        intent = read(job / 'intent.json') if (job / 'intent.json').exists() else {}
        if hypothesis_id and intent.get('hypothesis') != hypothesis_id:
            raise ValueError('Job ID already belongs to another hypothesis')
        if C.seal(job):
            C.reconcile(root, state)
            return job
    else:
        if hypothesis_id and not (root / 'hypotheses' / (C.identifier(hypothesis_id) + '.json')).exists():
            raise ValueError('Unknown hypothesis ID')
        practice = {json.loads(line)['id'] for path in (root / 'assets/practice').glob('*.jsonl')
                    for line in path.read_text().split('\n') if line.strip()}
        validate_request(req, practice)
        if state['used'] >= state['budget']:
            raise ValueError('Forward budget exhausted')
        stage = root / 'jobs' / ('.tmp-' + uuid.uuid4().hex)
        stage.mkdir()
        hashes = snapshot(root, stage / 'edits')
        write(stage / 'request.json', req)
        write(stage / 'edit_hashes.json', hashes)
        C.write(stage / 'intent.json', dict(id=name, status='pending', hypothesis=hypothesis_id,
                created=time.time(), asset_manifest_sha256=sha(root / 'assets/manifest.json')
                if (root / 'assets/manifest.json').exists() else None))
        stage.rename(job)
        C.event(root, 'forward_requested', job=name, hypothesis=hypothesis_id)
    if state['used'] >= state['budget']:
        raise ValueError('Forward budget exhausted')
    intent = read(job / 'intent.json')
    intent['status'] = 'running'
    C.write(job / 'intent.json', intent)
    try:
        gpu_job(root, state, job)
        if not C.seal(job):
            raise RuntimeError('GPU worker did not produce a complete result')
        C.event(root, 'forward_finished', job=name)
    except BaseException as exc:
        intent.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        C.write(job / 'intent.json', intent)
        C.event(root, 'forward_failed', job=name, error=str(exc))
        raise
    finally:
        C.reconcile(root, state)
        C.handoff(root)
    return job


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=REPO / '.sandbox')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('build')
    init = sub.add_parser('init')
    init.add_argument('--budget', type=int, choices=[3, 10, 20, 25, 50], default=50)
    init.add_argument('--python-budget', type=int, default=300)
    init.add_argument('--smoke', action='store_true', help='Infrastructure smoke run; permits a 3-pass budget')
    for name in ('prepare', 'start', 'stop', 'status', 'freeze', 'evaluate', 'handoff', 'history'):
        sub.add_parser(name)
    code = sub.add_parser('python')
    code.add_argument('file', type=Path)
    code.add_argument('--explain', required=True)
    code.add_argument('--id')
    forward = sub.add_parser('forward')
    forward.add_argument('request', type=Path)
    forward.add_argument('--id')
    forward.add_argument('--hypothesis')
    cp = sub.add_parser('checkpoint')
    cp.add_argument('--name')
    cp.add_argument('--note', default='')
    pause = sub.add_parser('pause')
    pause.add_argument('--note', default='')
    resume = sub.add_parser('resume')
    resume.add_argument('--checkpoint')
    resume.add_argument('--start', action='store_true')
    evidence = sub.add_parser('import-evidence')
    evidence.add_argument('directory', type=Path)
    hyp = sub.add_parser('hypothesis')
    hyp.add_argument('id')
    for arg in ('statement', 'prediction', 'parent', 'status', 'conclusion'):
        hyp.add_argument('--' + arg)
    export = sub.add_parser('export')
    export.add_argument('file', type=Path)
    export.add_argument('--include-weights', action='store_true')
    restore = sub.add_parser('restore')
    restore.add_argument('file', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    if a.command == 'restore':
        C.restore_bundle(a.file, root)
        with locked(root):
            C.reconcile(root, read(root / 'state.json'))
            C.handoff(root)
        print(root / 'handoff.md')
        return
    if a.command == 'build':
        for role, tag in [('editor', EDITOR), ('gpu', GPU)]:
            subprocess.run(['docker', 'build', '-f', str(REPO / f'experiments/Dockerfile.{role}'),
                            '-t', tag, str(REPO)], check=True)
        return
    if a.command == 'init':
        if a.python_budget < 1:
            raise ValueError('Python budget must be positive')
        if a.budget == 3 and not a.smoke:
            raise ValueError('A 3-pass budget requires --smoke')
        root.mkdir(parents=True, exist_ok=True)
        with locked(root):
            if (root / 'state.json').exists():
                raise ValueError('Run already exists; choose a new --root')
            for directory in ('assets', 'work/edited', 'jobs', 'code'):
                (root / directory).mkdir(parents=True, exist_ok=True)
            write(root / 'state.json', dict(budget=a.budget, used=0, python_runs=0, python_budget=a.python_budget,
                  frozen=False, smoke=a.smoke, container='wsp-' + uuid.uuid4().hex[:12],
                  editor_image=EDITOR, gpu_image=GPU))
            C.ensure(root)
            C.handoff(root)
        return
    with locked(root):
        state = C.reconcile(root, read(root / 'state.json'))
        if a.command == 'status':
            print(json.dumps(dict(state, experiment=C.ensure(root)), indent=2))
            return
        if a.command == 'prepare':
            job = root / 'jobs' / ('prepare-' + uuid.uuid4().hex[:8])
            job.mkdir()
            gpu_job(root, state, job, online=True)
        elif a.command == 'start':
            require_active(root, state)
            if not (root / 'assets/manifest.json').exists():
                raise ValueError('Assets not prepared')
            editor_start(root, state)
        elif a.command == 'stop':
            stop_editor(state)
        elif a.command == 'python':
            print(json.dumps(run_python(root, state, a.file.read_text(), a.explain, a.id)))
        elif a.command == 'forward':
            print(run_forward(root, state, read(a.request), a.id, a.hypothesis))
        elif a.command == 'hypothesis':
            require_active(root, state)
            print(json.dumps(C.hypothesis(root, a.id, a.statement, a.prediction, a.parent,
                                         a.status, a.conclusion), ensure_ascii=False, indent=2))
        elif a.command == 'import-evidence':
            print(C.import_evidence(root, a.directory))
        elif a.command == 'checkpoint':
            print(json.dumps(C.checkpoint(root, a.name, a.note), indent=2))
        elif a.command == 'pause':
            if state['frozen']:
                raise ValueError('Frozen experiment')
            stop_editor(state)
            cp = C.checkpoint(root, note=a.note)
            memory = C.ensure(root)
            memory.update(phase='paused', next_step=a.note, last_checkpoint=cp['id'])
            C.write(root / 'experiment.json', memory)
            C.event(root, 'pause', checkpoint=cp['id'])
            print(cp['id'])
        elif a.command == 'resume':
            if state['frozen']:
                raise ValueError('Frozen experiment')
            if a.checkpoint:
                stop_editor(state)
                C.restore_checkpoint(root, a.checkpoint)
            memory = C.ensure(root)
            memory['phase'] = 'active'
            C.write(root / 'experiment.json', memory)
            C.event(root, 'resume', checkpoint=a.checkpoint)
            if a.start:
                editor_start(root, state)
            print(root / 'handoff.md')
        elif a.command == 'handoff':
            print(root / 'handoff.md')
        elif a.command == 'history':
            for path in sorted((root / 'history').glob('*.json')):
                print(json.dumps(read(path), ensure_ascii=False))
        elif a.command == 'export':
            if C.ensure(root)['phase'] != 'paused':
                raise ValueError('Pause the experiment before exporting a consistent bundle')
            for job in (root / 'jobs').glob('forward-*'):
                if (job / 'container.json').exists() and not C.seal(job):
                    name = read(job / 'container.json')['name']
                    process = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', name],
                                             text=True, capture_output=True)
                    if process.returncode == 0 and process.stdout.strip() == 'true':
                        raise ValueError('GPU job is still running; resume that job ID and collect its results before exporting')
                    if process.returncode and 'No such' not in process.stderr:
                        raise RuntimeError('Cannot verify pending GPU container state; reconnect Docker before exporting')
            C.handoff(root)
            print(C.export_bundle(root, a.file, a.include_weights))
        elif a.command == 'freeze':
            freeze(root, state)
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
                    if C.seal(job):
                        print(job / 'output/scores.json')
                        continue
                    job.mkdir(exist_ok=True)
                    if not (job / 'request.json').exists():
                        shutil.copytree(root / 'submission', job / 'edits')
                        write(job / 'request.json', dict(type='generate', model=model, benchmark=bench,
                                                         n=n, max_tokens=32768))
                    gpu_job(root, state, job)
                    C.seal(job)
                    print(job / 'output/scores.json')
        C.handoff(root)


if __name__ == '__main__':
    main()
