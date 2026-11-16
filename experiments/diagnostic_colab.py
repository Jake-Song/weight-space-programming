"""Small independent baseline pilot: Colab GPU worker, headless Astra max planner.
No benchmark scores or published edits are shown to the editor. No local sandbox
forward budget is spent: this is a separately recorded infrastructure pilot.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import tempfile
import uuid

REPO = Path(__file__).resolve().parents[1]
MODEL = 'gpt-6-astra'
EFFORT = 'max'


def call(cmd, **kwargs):
    return subprocess.run(list(map(str, cmd)), check=True, **kwargs)


def remote(session, code, folder, label):
    path = folder / (label + '.remote.py')
    path.write_text(code)
    remote_path = '/content/wsp-driver-' + uuid.uuid4().hex[:12] + '.py'
    call(['colab', 'upload', '-s', session, path, remote_path])
    call(['colab', 'ssh', '-s', session, '--', 'python3', remote_path])


def astra(prompt, schema, folder, label):
    output = folder / (label + '.json')
    schema_path = folder / (label + '.schema.json')
    schema_path.write_text(json.dumps(schema))
    prompt_path = folder / (label + '.prompt.txt')
    prompt_path.write_text(prompt)
    # No repository/benchmark context, no user-configured MCPs or shell tools.
    cwd = Path(tempfile.mkdtemp(prefix='wsp-astra-'))
    cmd = ['codex', 'exec', '--ignore-user-config', '--ephemeral', '--skip-git-repo-check',
           '--sandbox', 'read-only', '--disable', 'shell_tool', '--disable', 'multi_agent',
           '-c', 'web_search="disabled"', '-c', f'model_reasoning_effort="{EFFORT}"',
           '-m', MODEL, '-C', str(cwd), '--output-schema', str(schema_path), '--json',
           '-o', str(output), '-']
    (folder / (label + '.invocation.json')).write_text(json.dumps(dict(model=MODEL, effort=EFFORT, command=cmd), indent=2))
    with (folder / (label + '.events.jsonl')).open('w') as stdout, (folder / (label + '.stderr.log')).open('w') as stderr:
        call(cmd, input=prompt, text=True, stdout=stdout, stderr=stderr, timeout=900)
    return json.loads(output.read_text())


def obj(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def verified_report(folder):
    try:
        from .diagnostic_metrics import measure
    except ImportError:
        from diagnostic_metrics import measure
    metrics = measure(folder)
    evidence = dict(plan=json.loads((folder/'plan.json').read_text()),
                    scores=json.loads((folder/'artifacts/output/scores.json').read_text()),
                    timing=json.loads((folder/'artifacts/timing.json').read_text()),
                    prospective_metrics=metrics)
    schema = obj(dict(hypothesis_assessment={'type': 'string'}, findings={'type': 'string'},
                      next_hypothesis={'type': 'string'}, limitations={'type': 'string'}))
    report = astra('Assess the original prospective hypothesis using these exact measured criteria. '
        'Do not use tools. Respond in Korean. This is eight selected training problems, one sample each, original weights only. '
        'Distinguish canonical boxed-answer accuracy from diagnostic accuracy requiring normal completion. '
        'A 30-token repetition flag is a literal repeated-span signal, not proof of a semantic loop. '
        'Do not claim weight localization, editing improvement, or generalization. State which prospective thresholds failed. '
        'Suggest one next weight-analysis hypothesis without importing any published edit. Evidence:\n' + json.dumps(evidence),
        schema, folder, 'verified_report')
    print(json.dumps(dict(metrics=metrics, report=report),ensure_ascii=False,indent=2),flush=True)
    return report


def finish_pilot(folder, meta):
    timing = json.loads((folder/'artifacts/timing.json').read_text())
    scores = json.loads((folder / 'artifacts/output/scores.json').read_text())
    records = [json.loads(line) for line in (folder / 'artifacts/output/original.jsonl').read_text().split(chr(10)) if line.strip()]
    observations = []
    for row in records:
        tokens = row['token_ids']
        grams = [tuple(tokens[i:i+20]) for i in range(max(0, len(tokens)-19))]
        repeat = 1 - len(set(grams))/len(grams) if grams else 0
        observations.append(dict(id=row['id'], finish_reason=row['finish_reason'], tokens=len(tokens),
                                 duplicate_20gram_fraction=repeat, excerpt=row['text'][-500:]))
    (folder / 'observations.json').write_text(json.dumps(observations, indent=2))
    report = json.loads((folder/'verified_report.json').read_text()) if (folder/'verified_report.json').exists() else verified_report(folder)
    meta.update(completed=time.time(), elapsed_seconds=timing['elapsed_seconds'], scores=scores)
    (folder / 'pilot.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(dict(scores=scores,timing=timing,report=report,folder=str(folder)),ensure_ascii=False,indent=2),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', default='wsp-diagnostic')
    p.add_argument('--output', type=Path)
    p.add_argument('--skip-setup', action='store_true')
    p.add_argument('--report-only', action='store_true')
    a = p.parse_args()
    folder = (a.output or REPO / '.sandbox/diagnostics' / ('pilot-' + uuid.uuid4().hex[:8])).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if a.report_only:
        verified_report(folder)
        return
    metadata_file = folder / 'pilot.json'
    meta = json.loads(metadata_file.read_text()) if metadata_file.exists() else dict(
        session=a.session, editor_model=MODEL, reasoning_effort=EFFORT,
        started=time.time(), protocol='independent baseline diagnostic; not a 50-pass edit run',
        sampling=dict(temperature=.6, top_p=.95, top_k=20, presence_penalty=1.5, max_tokens=32768))
    if meta.get('completed'):
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return
    timing_file = folder / 'artifacts/timing.json'
    if timing_file.exists() and json.loads(timing_file.read_text())['returncode'] == 0:
        # GPU results survived: finish locally even after the Colab VM has gone away.
        finish_pilot(folder, meta)
        return
    worker_hash = hashlib.sha256((REPO/'experiments/worker.py').read_bytes()).hexdigest()
    if meta.get('worker_sha256', worker_hash) != worker_hash:
        raise ValueError('Worker changed during an unfinished pilot; restore its original source or choose a new output folder')
    meta.update(session=a.session, worker_sha256=worker_hash)
    metadata_file.write_text(json.dumps(meta, indent=2))
    print('PILOT', folder, flush=True)
    if not a.skip_setup:
        remote(a.session, (REPO / 'scripts/setup_colab.py').read_text(), folder, 'setup')
    remote(a.session, "from pathlib import Path\nPath('/content/wsp-diagnostic').mkdir(exist_ok=True)\n", folder, 'mkdir')
    for src, dst in [(REPO / 'experiments/worker.py', 'worker.py'),
                     (REPO / 'results/vendor/code/mathqa.py', 'mathqa.py')]:
        call(['colab', 'upload', '-s', a.session, src, '/content/wsp-diagnostic/' + dst])
    remote(a.session, '''
import os, subprocess, sys
from pathlib import Path
root=Path('/content/wsp-diagnostic')
(root/'assets').mkdir(exist_ok=True)
(root/'prepare').mkdir(exist_ok=True)
env=dict(os.environ,WSP_ASSETS=str(root/'assets'),WSP_JOB=str(root/'prepare'))
if not (root/'assets/manifest.json').exists():
    result=subprocess.run([sys.executable,str(root/'worker.py'),'prepare'],env=env,capture_output=True,text=True)
    print(result.stdout[-3000:],flush=True)
    print(result.stderr[-6000:],flush=True)
    result.check_returncode()
import json, random
rows=[]
for name in ['gsm8k_train','math_train']:
    source=[json.loads(x) for x in (root/f'assets/practice/{name}.jsonl').read_text().split(chr(10)) if x.strip()]
    rows.extend(random.Random(20261116).sample(source,16))
(root/'practice_pool.json').write_text(json.dumps(rows))
print('ASSETS_READY',len(rows),flush=True)
''', folder, 'prepare')
    call(['colab', 'download', '-s', a.session, '/content/wsp-diagnostic/practice_pool.json', folder / 'practice_pool.json'])
    pool = json.loads((folder / 'practice_pool.json').read_text())
    selection = obj(dict(hypothesis={'type': 'string'}, explanation={'type': 'string'},
                         problems={'type': 'array', 'items': {'type': 'string', 'enum': [r['id'] for r in pool]}}))
    plan = json.loads((folder/'plan.json').read_text()) if (folder/'plan.json').exists() and (folder/'plan.json').stat().st_size else astra('You are the experiment editor. Do not use tools. Plan ONE small baseline diagnostic before a weight-editing experiment. '
        'Choose exactly 8 distinct practice IDs from the pool, four GSM8K and four MATH. No edits yet. '
        'The original Qwen3.5-0.8B will answer with thinking enabled, temperature .6, top_p .95, top_k20, presence penalty1.5, '
        '32768 tokens, one sample per problem. State a falsifiable diagnostic hypothesis about completion, accuracy and repetition. '
        'Do not infer results. This is training data, not test data. Return the requested JSON. Pool:\n' + json.dumps(pool), selection, folder, 'plan')
    ids = plan['problems']
    if len(ids) != 8 or len(set(ids)) != 8 or any(x not in {r['id'] for r in pool} for x in ids):
        raise ValueError('Invalid 8-problem diagnostic plan')
    gsm_ids = {r['id'] for r in pool[:16]}
    if sum(x in gsm_ids for x in ids) != 4:
        raise ValueError('Require four problems from each practice source')
    request = dict(type='generate', model='original', problems=ids, n=1, max_tokens=32768)
    (folder / 'request.json').write_text(json.dumps(request, indent=2))
    print('HYPOTHESIS', plan['hypothesis'], flush=True)
    remote_root = meta.get('remote_root') or '/content/wsp-diagnostic/pilot-' + uuid.uuid4().hex[:16]
    meta['remote_root'] = remote_root
    metadata_file.write_text(json.dumps(meta, indent=2))
    remote(a.session, f"from pathlib import Path\nPath({remote_root!r}).mkdir(exist_ok=True)\n", folder, 'job')
    call(['colab', 'upload', '-s', a.session, folder / 'request.json', remote_root + '/request.json'])
    remote(a.session, f'''
import os, subprocess, sys, json, time, tarfile
from pathlib import Path
job=Path({remote_root!r})
(job/'edits').mkdir(exist_ok=True)
started=time.time()
env=dict(os.environ,WSP_ASSETS='/content/wsp-diagnostic/assets',WSP_JOB=str(job),VLLM_ENABLE_V1_MULTIPROCESSING='1')
if (job/'worker.log').exists():
    import uuid
    (job/'worker.log').rename(job/('worker.attempt-'+uuid.uuid4().hex[:8]+'.log'))
with (job/'worker.log').open('w') as log:
    result=subprocess.run([sys.executable,'/content/wsp-diagnostic/worker.py','run'],env=env,stdout=log,stderr=subprocess.STDOUT)
(job/'timing.json').write_text(json.dumps(dict(elapsed_seconds=time.time()-started,returncode=result.returncode)))
with tarfile.open(str(job)+'.tar.gz','w:gz') as archive:
    for name in ['output','request.json','worker.log','timing.json']:
        path=job/name
        if path.exists(): archive.add(path,arcname=name)
print('GPU_RETURN',result.returncode,flush=True)
print((job/'worker.log').read_text()[-6000:],flush=True)
''', folder, 'generate')
    call(['colab', 'download', '-s', a.session, remote_root + '.tar.gz', folder / 'artifacts.tar.gz'])
    import tarfile
    with tarfile.open(folder / 'artifacts.tar.gz') as archive:
        archive.extractall(folder / 'artifacts', filter='data')
    timing = json.loads((folder / 'artifacts/timing.json').read_text())
    if timing['returncode']:
        raise RuntimeError(f'GPU worker failed; inspect {folder}/artifacts/worker.log')
    finish_pilot(folder, meta)


if __name__ == '__main__':
    main()
