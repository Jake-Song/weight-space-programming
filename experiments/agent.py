"""Run Astra(max) with 40-turn conversations until final submission."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time
import uuid

try:
    from . import continuity as C, runner as R
    from .diagnostic_colab import obj
except ImportError:
    import continuity as C
    import runner as R
    from diagnostic_colab import obj


SCHEMA = obj(dict(kind={'type':'string','enum':['python','forward','note','stop','submit']},
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
    state = C.read(root / 'state.json')
    data['runtime_limits'] = dict(editor_memory=state.get('editor_memory', '96g'),
                                  python_budget=state.get('python_budget', 300), forward_budget=state['budget'])
    if state.get('smoke'):
        data['smoke_instructions'] = ('This is a 3-pass infrastructure smoke test, not a capability experiment. '
            'Use small requests: at most 2 practice items, n=1, max_tokens=512 for generation; '
            'at most 2 short texts for teacher forcing. Exercise Python persistence, original generation, '
            'teacher-forced activation inspection, a justified tiny direct edit and a paired generation. '
            'Use exactly three successful forward calls, then submit. Aim for at most 12 actions. '
            'Do not claim improved capability from smoke scores. The CPU kernel memory cap is 5 GiB.')
    practice = []
    for path in sorted((root / 'assets/practice').glob('*.jsonl')):
        rows = [json.loads(line) for line in path.read_text().split('\n') if line.strip()]
        practice.extend(dict(id=r['id'], problem=r['problem'], source=r.get('source')) for r in rows[:8])
    return '''TASK: Make Qwen3.5-0.8B better at mathematics by editing its weights directly.
Diagnose a failure mechanism, identify responsible components and repair it with justified changes.
The text model has 24 layers of width 1024, SwiGLU MLP width 3584, and full-attention
layers 3, 7, 11, 15, 19, 23; the remaining layers use Gated DeltaNet linear attention.
Practice files contain GSM8K train (7469), MATH train (7275) and AIME (714) problems.
Only practice data may guide edits. Hidden final evaluation includes MATH-500 and GSM8K.
Do not force thinking to close early regardless of the problem; repair specific failures within thinking.
Continue ONE action in this persistent weight-space experiment. You are gpt-6-astra at max effort.
Use only the recorded context below; do not use tools, web, prior published edits or benchmark data.
Your response is a structured action, not host shell commands. Available kinds:
python: code runs ONLY in an isolated CPU kernel with NAMES, load(name), load_original(name),
save_edit(name, float32_tensor), drop_edit(name), edited(), np, torch, tok; /practice read-only, /work writable.
tok is a tokenizers.Tokenizer, not a Transformers tokenizer: tok.encode(text).ids and tok.decode(ids)
are available; apply_chat_template is not. Read /model/tokenizer_config.json for the template if needed.
Kernel variables can be lost between sessions: load inputs explicitly and persist useful arrays with numpy,
never pickle. No model forward in the kernel; no training, backward, autograd, optimizers or iterative loss fitting.
Only transformer block weights can change; embeddings, output, final norm and whole-tensor rescaling are forbidden.
forward: request_json is JSON for the trusted GPU tool: type generate (model original/edited/both,
problems practice IDs, n>=1, max_tokens<=32768; <=512 total generations, both <=256 per arm), or teacher_forced
(model original/edited, texts<=1024, return token_logprobs/hidden_states[0..24]/mlp_act[0..23]).
Use no benchmark IDs. Full results are written to /work/forward-<action-id>/; inspect them with python.
note: record a hypothesis assessment, notes and the next step without spending forward or Python budget.
stop: a legacy no-code response; it does NOT stop the experiment before submission.
submit: after ALL forward passes are spent, freeze the current edits as the final submission.
Before submitting, use Python if needed to restore your best edit. Explain the final change and evidence.
The run continues until all forward passes are used; a no-code reply cannot end it early.
Python executions have a separate limit given by runtime_limits.python_budget. Failed forward requests do not spend a pass.
Keep /work/notes.md current and keep copies of promising edits.
Diagnose failures, localize causally, vary intervention strength, and confirm on separate practice problems.
Use paired original/edited measurements on the SAME problems; retain improvements exceeding their
standard error on problems not used to select the edit. Reduced truncation alone is not success.
The evaluation uses thinking on, temperature .6, top_p .95, top_k 20, presence penalty 1.5,
and 32768 new tokens; config/tokenizer/template remain fixed. Final test scores are never returned.
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
    if any(not isinstance(v,str) for v in decision.values()) or decision['kind'] not in ('python','forward','note','stop','submit'):
        raise ValueError('Invalid action')
    if decision['kind'] in ('python','forward') and (not decision['hypothesis_id'] or not decision['explanation'].strip()):
        raise ValueError('Execution requires a hypothesis and explanation')
    if decision['verdict'] != 'none' and (not decision['hypothesis_id'] or not decision['conclusion'].strip()):
        raise ValueError('A hypothesis verdict requires an ID and a conclusion')
    if decision['kind'] == 'submit' and not decision['explanation'].strip():
        raise ValueError('Submission requires a final explanation')
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
    if decision['kind'] == 'submit' and state['used'] != state['budget']:
        result = dict(error='Use all forward passes before submitting; continue the experiment.')
        C.write(action / 'result.json', result)
        metadata.update(status='complete', completed=time.time())
        C.write(action / 'state.json', metadata)
        return 'note'
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
    if decision['kind'] == 'submit':
        R.freeze(root, state)
        (root / 'final_answer.md').write_text(decision['explanation'] + '\n' + decision['notes'] + '\n')
        memory['phase'] = 'submitted'
    C.write(root / 'experiment.json', memory)
    metadata.update(status='complete', completed=time.time())
    C.write(action / 'state.json', metadata)
    C.event(root, 'agent_action', action=name, kind_of_action=decision['kind'])
    C.handoff(root)
    return decision['kind']


def require_driver_active(root, state):
    # Finish postprocessing after freeze persisted but the action receipt did not.
    if state['frozen']:
        pending = [p for p in (root / 'agent').glob('action-*')
                   if C.read(p / 'state.json')['status'] == 'running']
        if len(pending) == 1 and C.read(pending[0] / 'decision.json')['kind'] == 'submit':
            return
    R.require_active(root, state)


def run_steps(root, limit=None, planner=None):
    root = Path(root).resolve()
    if limit is not None and limit < 1:
        raise ValueError('--steps must be positive')
    if planner is None:
        try:
            from .conversation import Conversation
        except ImportError:
            from conversation import Conversation
        planner = Conversation(root)
    with (root / '.agent.lock').open('a') as driver_lock:
        fcntl.flock(driver_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        count = 0
        while limit is None or count < limit:
            count += 1
            with R.locked(root):
                state = C.reconcile(root, C.read(root / 'state.json'))
                require_driver_active(root, state)
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
                prompt = None
                if need_plan:
                    if getattr(planner, 'needs_handoff', True):
                        prompt = decision_prompt(root)
                    else:
                        previous = [p for p in actions if p != action and (p / 'result.json').exists()]
                        result = C.read(previous[-1] / 'result.json') if previous else {}
                        prompt = ('Previous action result:\n' + json.dumps(result, ensure_ascii=False)
                                  + '\nBudget: ' + json.dumps(dict(used=state['used'], budget=state['budget'],
                                                               python_runs=state['python_runs'],
                                                               python_budget=state.get('python_budget', 300)))
                                  + '\nChoose the next action. Submit only after all passes are used.')
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
                require_driver_active(root, state)
                try:
                    kind = execute(root, action, state)
                except BaseException as exc:
                    metadata = C.read(action / 'state.json')
                    metadata['last_error'] = str(exc)
                    C.write(action / 'state.json', metadata)
                    C.handoff(root)
                    raise
                print(action.name + ': ' + kind, flush=True)
                if kind == 'submit':
                    break
        with R.locked(root):
            C.checkpoint(root, note=f'End of agent invocation ({count} actions)')
            C.handoff(root)
    return root / 'handoff.md'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=R.REPO / '.sandbox')
    p.add_argument('--steps', type=int, help='Optional action cap; default: continue until submission')
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
