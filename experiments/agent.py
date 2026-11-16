"""Run a bounded number of headless Astra(max) experiment actions, then resume later."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time
import uuid

try:
    from . import continuity as C, runner as R
    from .diagnostic_colab import astra, obj
except ImportError:
    import continuity as C
    import runner as R
    from diagnostic_colab import astra, obj


SCHEMA = obj(dict(kind={'type':'string','enum':['python','forward','note','stop']},
    hypothesis_id={'type':'string'}, statement={'type':'string'}, prediction={'type':'string'},
    parent={'type':'string'}, explanation={'type':'string'}, code={'type':'string'},
    request_json={'type':'string'}, notes={'type':'string'}, next_step={'type':'string'},
    verdict={'type':'string','enum':['none','supported','rejected','inconclusive']},
    conclusion={'type':'string'}))


def compact(data):
    result = dict(data)
    jobs = []
    for job in data['practice_jobs']:
        request = dict(job['request'])
        for key in ('texts', 'prompts', 'problems'):
            if key in request:
                values = request.pop(key)
                request[key + '_count'] = len(values)
                if key == 'problems':
                    request[key + '_preview'] = values[:16]
        jobs.append(dict(job, request=request))
    result['practice_jobs'] = jobs
    result['checkpoints'] = [{key:cp[key] for key in ('id','note','used','python_runs')} for cp in data['checkpoints']]
    result['notes'] = result['notes'][-8000:]
    result['python_actions'] = [dict(action, code=action['code'][:6000],
        result=dict(action['result'], output=action['result'].get('output','')[-3000:])
        if action['result'] else None) for action in data['python_actions'][-6:]]
    return result


def context_hash(root):
    data = C.context(root)
    data.pop('agent_actions', None)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def decision_prompt(root):
    data = compact(C.context(root))
    practice = []
    for path in sorted((root / 'assets/practice').glob('*.jsonl')):
        rows = [json.loads(line) for line in path.read_text().split('\n') if line.strip()]
        practice.extend(dict(id=r['id'], problem=r['problem'], source=r.get('source')) for r in rows[:8])
    return '''Continue ONE action in this persistent weight-space experiment. You are gpt-6-astra at max effort.
Use only the recorded context below; do not use tools, web, prior published edits or benchmark data.
Your response is a structured action, not host shell commands. Available kinds:
python: code runs ONLY in an isolated CPU kernel with NAMES, load(name), load_original(name),
save_edit(name, float32_tensor), drop_edit(name), edited(), np, torch, tok; /practice read-only, /work writable.
Kernel variables can be lost between sessions: load inputs explicitly and persist useful arrays with numpy,
never pickle. No model forward in the kernel; no training, backward, autograd, optimizers or iterative loss fitting.
Only transformer block weights can change; embeddings, output, final norm and whole-tensor rescaling are forbidden.
forward: request_json is JSON for the trusted GPU tool: type generate (model original/edited/both,
problems practice IDs, n>=1, max_tokens<=32768; <=512 total generations, both <=256 per arm), or teacher_forced
(model original/edited, texts<=1024, return token_logprobs/hidden_states[0..24]/mlp_act[0..23]).
Use no benchmark IDs. Full results are written to /work/forward-<action-id>/; inspect them with python.
note: record a hypothesis assessment, notes and the next step without spending forward or Python budget.
stop: checkpoint and pause this run. Do not declare success without evidence.
Every python/forward action must have an explanation and a hypothesis_id. A new hypothesis requires statement
and a falsifiable prediction, optionally parent ID; reuse an existing ID with empty statement/prediction.
verdict none means no assessment; otherwise provide a conclusion grounded in recorded job evidence.
Use empty strings for irrelevant fields. Treat imported prior pilots as evidence, not new calls in this budget.
Choose meaningful steps that diagnose, localize, intervene and verify on separate practice problems.
Do not spend the entire budget at once: only the next action will execute. Report notes/next_step in Korean.
Context:\n''' + json.dumps(data, ensure_ascii=False) + '\nPractice ID examples:\n' + json.dumps(practice, ensure_ascii=False)


def validate_decision(decision):
    if set(decision) != set(SCHEMA['properties']):
        raise ValueError('Invalid action fields')
    if any(not isinstance(v,str) for v in decision.values()) or decision['kind'] not in ('python','forward','note','stop'):
        raise ValueError('Invalid action')
    if decision['kind'] in ('python','forward') and (not decision['hypothesis_id'] or not decision['explanation'].strip()):
        raise ValueError('Execution requires a hypothesis and explanation')
    if decision['verdict'] != 'none' and (not decision['hypothesis_id'] or not decision['conclusion'].strip()):
        raise ValueError('A hypothesis verdict requires an ID and a conclusion')
    if decision['kind'] == 'python' and not decision['code'].strip():
        raise ValueError('Python action requires code')
    if decision['kind'] == 'forward':
        if not isinstance(json.loads(decision['request_json']), dict):
            raise ValueError('Forward request must be a JSON object')
    return decision


def execute(root, action, state):
    decision = validate_decision(C.read(action / 'decision.json'))
    name = action.name
    metadata = C.read(action / 'state.json')
    if metadata['status'] == 'proposed':
        if context_hash(root) != metadata['context_sha256']:
            raise RuntimeError('Experiment changed after planning; abandon this action and plan again')
        cp = C.checkpoint(root, note='Before agent action ' + name)
        metadata.update(status='running', checkpoint=cp['id'])
        C.write(action / 'state.json', metadata)
    hyp = decision['hypothesis_id']
    if hyp:
        C.hypothesis(root, hyp, decision['statement'] or None, decision['prediction'] or None,
                     decision['parent'] or None,
                     decision['verdict'] if decision['verdict'] != 'none' else None,
                     decision['conclusion'] or None)
    if decision['kind'] == 'python':
        R.editor_start(root, state)
        result = R.run_python(root, state, decision['code'], decision['explanation'], name)
    elif decision['kind'] == 'forward':
        job = R.run_forward(root, state, json.loads(decision['request_json']), name, hyp)
        result = dict(job=job.name, scores=C.read(job / 'output/scores.json'))
    else:
        result = dict(note=decision['notes'])
    # Result is persisted before postprocessing; retries reuse the code/job receipt.
    C.write(action / 'result.json', result)
    notes = root / 'work/notes.md'
    if notes.is_symlink():
        raise ValueError('Notebook must not be a symlink')
    text = notes.read_text() if notes.exists() else ''
    marker = '<!-- ' + name + ' -->'
    if marker not in text:
        with notes.open('a') as f:
            f.write('\n' + marker + '\n' + decision['notes'] + '\n')
    memory = C.ensure(root)
    memory['next_step'] = decision['next_step']
    if decision['kind'] == 'stop':
        R.stop_editor(state)
        memory['phase'] = 'paused'
    C.write(root / 'experiment.json', memory)
    metadata.update(status='complete', completed=time.time())
    C.write(action / 'state.json', metadata)
    C.event(root, 'agent_action', action=name, kind_of_action=decision['kind'])
    C.handoff(root)
    return decision['kind']


def run_steps(root, limit, planner=astra):
    root = Path(root).resolve()
    if limit < 1:
        raise ValueError('--steps must be positive')
    with (root / '.agent.lock').open('a') as driver_lock:
        fcntl.flock(driver_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for _ in range(limit):
            with R.locked(root):
                state = C.reconcile(root, C.read(root / 'state.json'))
                R.require_active(root, state)
                C.handoff(root)
                actions = sorted((root / 'agent').glob('action-*'))
                pending = [p for p in actions if C.read(p / 'state.json')['status'] not in ('complete','abandoned')]
                if len(pending) > 1:
                    raise ValueError('Multiple pending actions; inspect history before continuing')
                if pending:
                    action = pending[0]
                else:
                    action = root / 'agent' / f'action-{len(actions)+1:06d}'
                    action.mkdir()
                    C.write(action / 'state.json', dict(status='planning', context_sha256=context_hash(root), created=time.time()))
                need_plan = not (action / 'decision.json').exists()
                metadata = C.read(action / 'state.json')
                if need_plan:
                    metadata['context_sha256'] = context_hash(root)
                    C.write(action / 'state.json', metadata)
                elif metadata['status'] == 'planning':
                    # Decision was atomically saved just before a controller interruption.
                    metadata['status'] = 'proposed'
                    C.write(action / 'state.json', metadata)
                prompt = decision_prompt(root) if need_plan else None
            if need_plan:
                attempt = action / ('attempt-' + uuid.uuid4().hex[:8])
                attempt.mkdir()
                decision = planner(prompt, SCHEMA, attempt, 'decision')
                validate_decision(decision)
                C.write(action / 'decision.json', decision)
                metadata = C.read(action / 'state.json')
                metadata['status'] = 'proposed'
                C.write(action / 'state.json', metadata)
            with R.locked(root):
                state = C.reconcile(root, C.read(root / 'state.json'))
                R.require_active(root, state)
                try:
                    kind = execute(root, action, state)
                except BaseException as exc:
                    metadata = C.read(action / 'state.json')
                    metadata['last_error'] = str(exc)
                    C.write(action / 'state.json', metadata)
                    C.handoff(root)
                    raise
                print(action.name + ': ' + kind, flush=True)
                if kind == 'stop':
                    break
        with R.locked(root):
            C.checkpoint(root, note=f'End of bounded agent invocation (up to {limit} actions)')
            C.handoff(root)
    return root / 'handoff.md'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=R.REPO / '.sandbox')
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--abandon')
    p.add_argument('--reason')
    a = p.parse_args()
    root = a.root.resolve()
    if a.abandon:
        if not a.reason:
            p.error('--abandon requires --reason')
        with (root / '.agent.lock').open('a') as driver_lock:
            fcntl.flock(driver_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with R.locked(root):
                path = root / 'agent' / C.identifier(a.abandon) / 'state.json'
                metadata = C.read(path)
                if metadata['status'] in ('complete','abandoned'):
                    raise ValueError('Action is already finalized')
                metadata.update(status='abandoned', reason=a.reason)
                C.write(path, metadata)
                C.event(root, 'abandon_action', action=a.abandon, reason=a.reason)
                C.handoff(root)
        return
    print(run_steps(root, a.steps))


if __name__ == '__main__':
    main()
