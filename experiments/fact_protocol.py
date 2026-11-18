"""Trusted host orchestration for candidate selection and held-out evaluation."""
import json
import shutil
import tempfile
from pathlib import Path

try:
    from . import continuity as C, fact_update_ko as fact
except ImportError:
    import continuity as C
    import fact_update_ko as fact

TASK = fact.TASK
TASKS = (TASK, 'fact_update_en')


def exploration_jobs(root):
    return [p for p in sorted((root / 'jobs').glob('forward-*')) if C.seal(p)]


def validate_exploration(root, state, req, validate, hypothesis):
    used = state['used']
    if used >= 16:
        raise ValueError('All exploration passes spent; freeze to select candidates')
    if used in (1, 2, 3):
        if req.get('type') != 'teacher_forced' or req.get('model') != 'original':
            raise ValueError('Passes 2–4 diagnose original activations')
        ret = req.get('return', {})
        if not isinstance(ret, dict) or not (ret.get('mlp_act') or ret.get('hidden_states')):
            raise ValueError('Expected return object, e.g. {"hidden_states":[4,8,12,16,20,24]}')
        return validate(req, set())
    arm = 'original' if used == 0 else 'edited'
    if req != dict(type='score_choices', model=arm):
        raise ValueError(f'This pass requires score_choices on {arm} with the full exploration split')
    if used and not hypothesis:
        raise ValueError('Link each candidate to a recorded hypothesis')


def validate_candidate(root, state, hashes):
    if not hashes:
        raise ValueError('Expected a nonempty candidate edit')
    if state['used'] >= 14:
        distinct = {tuple(sorted(C.read(p / 'edit_hashes.json').items()))
                    for p in exploration_jobs(root) if C.read(p / 'request.json')['model'] == 'edited'}
        distinct.add(tuple(sorted(hashes.items())))
        if len(distinct) + 15 - state['used'] < 3:
            raise ValueError('Submit a distinct candidate to fill three shortlist slots')


def rank(job, arm):
    score = C.read(job / 'output/scores.json')[arm]['pair_accuracy']
    audit = C.read(job / 'output/edit_audit.json')
    return (-score, sum(v['delta_l2_sq'] for v in audit), job.name)


def shortlist(root):
    jobs = [p for p in exploration_jobs(root) if C.read(p / 'request.json')['model'] == 'edited']
    seen, selected = set(), []
    for job in sorted(jobs, key=lambda p: rank(p, 'edited')):
        hashes = tuple(sorted(C.read(job / 'edit_hashes.json').items()))
        if hashes not in seen:
            selected.append(job.name)
            seen.add(hashes)
    if len(selected) < 3:
        raise ValueError('Three distinct candidate snapshots required')
    return selected[:3]


def fixed_job(root, state, name, req, source, gpu_job):
    job = root / 'jobs' / name
    if not job.exists():
        with tempfile.TemporaryDirectory(dir=job.parent) as tmp:
            stage = Path(tmp) / 'job'
            stage.mkdir()
            if source is None:
                (stage / 'edits').mkdir()
            else:
                shutil.copytree(source, stage / 'edits')
            C.write(stage / 'request.json', req)
            stage.rename(job)
    if C.read(job / 'request.json') != req or C.files(job / 'edits') != (C.files(source) if source else {}):
        raise ValueError('Fixed job inputs changed')
    if not C.seal(job):
        gpu_job(root, state, job)
        if not C.seal(job):
            raise RuntimeError('Incomplete fact-update job')
    return job


def select(root, state, gpu_job, freeze):
    if state.get('ceiling_reached') or state['used'] < 16:
        raise ValueError('Selection requires 16 completed exploration passes below ceiling')
    path = root / 'fact-candidates.json'
    selected = shortlist(root)
    if path.exists() and selected != C.read(path):
        raise ValueError('Shortlist changed after lock')
    C.write(path, selected)
    scored = []
    for i, source in enumerate([None, *selected]):
        arm = 'edited' if source else 'original'
        snapshot = root / 'jobs' / source if source else None
        if snapshot and C.files(snapshot / 'edits') != C.read(snapshot / 'edit_hashes.json'):
            raise ValueError('Candidate changed after scoring')
        job = fixed_job(root, state, f'selection-{i}', dict(type='score_choices', model=arm,
                        split='selection'), snapshot / 'edits' if source else None, gpu_job)
        if source:
            scored.append((rank(job, arm), snapshot))
    winner = min(scored)[1]
    C.reconcile(root, state)
    freeze(root, state, selected=winner)
    C.write(root / 'fact-selection.json', dict(winner=winner.name, candidates=selected))


def evaluate(root, state, gpu_job):
    arms = {}
    for label in ['original', 'edited', *(f'random-{i}' for i in range(5))]:
        arm = 'original' if label == 'original' else 'edited'
        req = dict(type='score_choices', model=arm, split='final')
        if label.startswith('random-'):
            req['random_seed'] = int(label.split('-')[1])
        job = fixed_job(root, state, 'eval-fact-'+label, req,
                        root / 'submission' if arm == 'edited' else None, gpu_job)
        arms[label] = C.read(job / f'output/{arm}_choices.json')
    dest = root / 'fact-report.json'
    C.write(dest, fact.report(arms))
    return dest
