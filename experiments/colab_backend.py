"""Colab GPU transport for the persistent local Docker editor.

Only this trusted worker runs remotely. Editor Python stays in the CPU container.
Session allocation and shutdown are explicit caller responsibilities.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def call(*args):
    subprocess.run([str(x) for x in args], check=True)


def download(session, remote_path, local_path):
    # The CLI contents API base64-decodes the entire response in RAM. Stream
    # large model/activation artifacts through SSH instead.
    local_path = Path(local_path)
    temporary = local_path.with_name(local_path.name + '.partial')
    with temporary.open('wb') as output:
        subprocess.run(['colab', 'ssh', '-s', session, '--', 'cat', remote_path],
                       stdout=output, check=True)
    temporary.replace(local_path)


def upload(session, local_path, remote_path):
    code = ('import sys,shutil; '
            'out=open(sys.argv[1],"wb"); '
            'shutil.copyfileobj(sys.stdin.buffer,out,1024*1024); out.close()')
    with Path(local_path).open('rb') as source:
        subprocess.run(['colab', 'ssh', '-s', session, '--', 'python3', '-c',
                        code, remote_path], stdin=source, check=True)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def remote_job(base, name, prepare=False):
    base = Path(base)
    job = base / 'jobs' / name
    job.mkdir(parents=True, exist_ok=True)
    with (job / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assets = base / 'assets'
        assets.mkdir(exist_ok=True)
        if not prepare:
            archive = job / 'input.tar'
            digest = sha(archive)
            receipt = job / 'input.sha256'
            if receipt.exists() and receipt.read_text() != digest:
                raise ValueError('Remote job input changed')
            if not receipt.exists():
                with tarfile.open(archive) as incoming:
                    incoming.extractall(job, filter='data')
                receipt.write_text(digest)
        env = dict(os.environ, WSP_ASSETS=str(assets), WSP_JOB=str(job),
                   VLLM_ENABLE_V1_MULTIPROCESSING='1')
        if not prepare:
            env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        start = time.time()
        with (job / 'worker.log').open('a') as log:
            result = subprocess.run([sys.executable, str(base / 'worker.py'),
                                     'prepare' if prepare else 'run'],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        (job / 'timing.json').write_text(json.dumps(dict(
            returncode=result.returncode, elapsed_seconds=time.time()-start)))
        with tarfile.open(str(job) + '.tar.gz', 'w:gz', compresslevel=1) as archive:
            for name in ('output', 'worker.log', 'timing.json'):
                if (job / name).exists():
                    archive.add(job / name, arcname=name)
        if prepare and result.returncode == 0:
            manifest = json.loads((assets / 'manifest.json').read_text())
            with tarfile.open(base / 'assets.tar', 'w') as archive:
                for rel in ['manifest.json', *manifest['files']]:
                    archive.add(assets / rel, arcname=rel)
        print((job / 'worker.log').read_text()[-6000:], flush=True)
        print('WORKER_RETURN', result.returncode, flush=True)


def gpu_job(root, state, job, online=False):
    session = state['colab_session']
    base = state['colab_root']
    remote = base + '/jobs/' + job.name
    call('colab', 'ssh', '-s', session, '--', 'mkdir', '-p', remote)
    if not online:
        archive = job / 'input.tar'
        if not archive.exists():
            with tarfile.open(archive, 'w') as outgoing:
                outgoing.add(job / 'request.json', arcname='request.json')
                outgoing.add(job / 'edits', arcname='edits')
        upload(session, archive, remote + '/input.tar')
    (job / 'runtime.json').write_text(json.dumps(dict(
        backend='colab', session=session, remote_job=remote,
        worker_sha256=state['colab_worker_sha256'], started=time.time()), indent=2))
    command = ['colab', 'ssh', '-s', session, '--', 'python3', base + '/colab_backend.py',
               'remote', '--root', base, '--job', job.name]
    if online:
        command.append('--prepare')
    call(*command)
    archive = job / 'remote-results.tar.gz'
    download(session, remote + '.tar.gz', archive)
    with tarfile.open(archive) as incoming:
        incoming.extractall(job, filter='data')
    timing = json.loads((job / 'timing.json').read_text())
    if timing['returncode']:
        raise RuntimeError(f'Colab worker failed; inspect {job}/worker.log')
    if not online:
        provenance = json.loads((job / 'output/provenance.json').read_text())
        error = None
        if provenance['input_identity']['worker_sha256'] != state['colab_worker_sha256']:
            error = 'Remote worker changed'
        if provenance['asset_manifest_sha256'] != sha(root / 'assets/manifest.json'):
            error = 'Remote and local assets differ'
        if error:
            (job / 'output').rename(job / f'rejected-output-{time.time_ns()}')
            raise ValueError(error)
    if online:
        manifest_path = root / 'assets/manifest.json'
        if manifest_path.exists():
            remote_manifest = job / 'remote-manifest.json'
            download(session, base + '/assets/manifest.json', remote_manifest)
            if sha(remote_manifest) != sha(manifest_path):
                raise ValueError('Prepared assets differ from existing experiment')
            return
        archive = job / 'assets.tar'
        download(session, base + '/assets.tar', archive)
        with tarfile.open(archive) as incoming:
            incoming.extractall(root / 'assets', filter='data')
        manifest = json.loads((root / 'assets/manifest.json').read_text())
        for rel, expected in manifest['files'].items():
            if sha(root / 'assets' / rel) != expected:
                raise ValueError(f'Downloaded asset changed: {rel}')
        archive.unlink()


def setup(root, session):
    from . import runner as R, continuity as C
    with R.locked(root):
        state = C.read(root / 'state.json')
        base = '/content/wsp-' + C.ensure(root)['id']
        worker_hash = sha(R.REPO / 'experiments/worker.py')
        if state.get('colab_worker_sha256', worker_hash) != worker_hash:
            raise ValueError('Worker changed; restore its original source')
        state.update(colab_session=session, colab_root=base, colab_worker_sha256=worker_hash)
        C.write(root / 'state.json', state)
        call('colab', 'ssh', '-s', session, '--', 'mkdir', '-p', base)
        for src, target in [('experiments/worker.py', 'worker.py'),
                            ('experiments/colab_backend.py', 'colab_backend.py'),
                            ('vendor/mathqa.py', 'mathqa.py'),
                            ('scripts/setup_colab.py', 'setup.py')]:
            call('colab', 'upload', '-s', session, R.REPO / src, base + '/' + target)
        call('colab', 'ssh', '-s', session, '--', 'python3', base + '/setup.py')
        # A replacement VM receives the original data snapshot and manifest;
        # prepare() downloads only missing pinned weights and verifies hashes.
        manifest_path = root / 'assets/manifest.json'
        if manifest_path.exists():
            manifest = C.read(manifest_path)
            for rel, expected in manifest['files'].items():
                if sha(root / 'assets' / rel) != expected:
                    raise ValueError(f'Local asset changed: {rel}')
            archive = root / 'assets-restore.tar'
            with tarfile.open(archive, 'w') as outgoing:
                for rel in ['manifest.json', *manifest['files']]:
                    if not rel.endswith('.safetensors'):
                        outgoing.add(root / 'assets' / rel, arcname=rel)
            upload(session, archive, base + '/assets-restore.tar')
            code = ('import pathlib,sys,tarfile; p=pathlib.Path(sys.argv[1]); '
                    '(p/"assets").mkdir(exist_ok=True); '
                    'tarfile.open(p/"assets-restore.tar").extractall(p/"assets",filter="data")')
            call('colab', 'ssh', '-s', session, '--', 'python3', '-c', code, base)
            archive.unlink()
        job = root / 'jobs/prepare-colab'
        job.mkdir(exist_ok=True)
        gpu_job(root, state, job, online=True)
        C.handoff(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['setup', 'remote'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--session')
    parser.add_argument('--job')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.command == 'remote':
        if not args.job or Path(args.job).name != args.job:
            parser.error('A single job directory name is required')
        remote_job(args.root, args.job, args.prepare)
    else:
        if not args.session:
            parser.error('--session is required')
        setup(args.root.resolve(), args.session)
