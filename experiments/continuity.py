"""Durable experiment memory. No model code or editor pickle is loaded here."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import time
import uuid

ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z')


def identifier(value):
    if not ID.fullmatch(value):
        raise ValueError('ID must contain 1..80 letters, digits, dots, underscores or hyphens')
    return value


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def files(directory):
    if directory.is_symlink():
        raise ValueError('Artifact directory cannot be a symlink')
    result = {}
    for p in sorted(directory.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'Symlinks cannot be checkpointed: {p}')
        if p.is_file():
            result[str(p.relative_to(directory))] = digest(p)
    return result


def event(root, kind, **fields):
    write(root / 'history' / f'{time.time_ns()}-{uuid.uuid4().hex[:8]}.json',
          dict(kind=kind, time=time.time(), **fields))


def ensure(root):
    path = root / 'experiment.json'
    if not path.exists():
        write(path, dict(id=uuid.uuid4().hex, created=time.time(), phase='active',
                         editor_model='gpt-6-astra', reasoning_effort='max', next_step=''))
    for name in ('history', 'hypotheses', 'checkpoints', 'evidence', 'agent'):
        (root / name).mkdir(exist_ok=True)
    return read(path)


def hypothesis(root, name, statement=None, prediction=None, parent=None,
               status=None, conclusion=None):
    path = root / 'hypotheses' / (identifier(name) + '.json')
    if path.exists():
        item = read(path)
        if statement is not None and statement != item['statement']:
            raise ValueError('Hypothesis statements are immutable; use a new ID')
        if prediction is not None and prediction != item['prediction']:
            raise ValueError('Predictions are immutable; use a new ID')
    else:
        if not statement or not prediction:
            raise ValueError('New hypotheses require --statement and --prediction')
        if parent and not (root / 'hypotheses' / (identifier(parent) + '.json')).exists():
            raise ValueError('Unknown parent hypothesis')
        item = dict(id=name, statement=statement, prediction=prediction,
                    parent=parent, status='proposed', conclusion='', created=time.time())
    if status:
        if status not in ('proposed', 'supported', 'rejected', 'inconclusive'):
            raise ValueError('Invalid hypothesis status')
        if status != 'proposed' and not conclusion:
            raise ValueError('A conclusion is required')
        item.update(status=status, conclusion=conclusion or '')
    write(path, item)
    event(root, 'hypothesis', hypothesis=item)
    return item


def seal(job):
    """Seal trusted outputs; an existing seal detects subsequent corruption."""
    output = job / 'output'
    if not (output / 'provenance.json').exists():
        return False
    if not (output / 'scores.json').exists():
        raise ValueError(f'Incomplete success marker: {job}')
    actual = files(output)
    path = job / 'result_manifest.json'
    if path.exists() and read(path) != actual:
        raise ValueError(f'Completed outputs changed: {job}')
    if not path.exists():
        write(path, actual)
    return True


def reconcile(root, state):
    """Accounting derives from successful jobs, independent of sessions or cache."""
    completed = 0
    for job in sorted((root / 'jobs').glob('forward-*')):
        if seal(job):
            completed += 1
            intent = read(job / 'intent.json') if (job / 'intent.json').exists() else {}
            intent.update(status='complete')
            write(job / 'intent.json', intent)
            destination = root / 'work' / job.name
            # Work copies are disposable views; trusted outputs remain outside /work.
            if not destination.exists():
                shutil.copytree(job / 'output', destination)
    state['used'] = completed
    write(root / 'state.json', state)
    ensure(root)
    return state


def checkpoint(root, name=None, note=''):
    name = identifier(name or ('cp-' + uuid.uuid4().hex[:12]))
    destination = root / 'checkpoints' / name
    if destination.exists():
        raise ValueError('Checkpoint already exists')
    stage = root / 'checkpoints' / ('.tmp-' + uuid.uuid4().hex)
    stage.mkdir()
    edits = root / 'work/edited'
    files(edits)  # reject links before copying
    shutil.copytree(edits, stage / 'edited')
    notes = root / 'work/notes.md'
    if notes.exists():
        if notes.is_symlink():
            raise ValueError('Notes must be a regular file')
        shutil.copyfile(notes, stage / 'notes.md')
    state = read(root / 'state.json')
    manifest = dict(id=name, created=time.time(), note=note, next_step=ensure(root).get('next_step',''), budget=state['budget'],
                    used=state['used'], python_runs=state['python_runs'], files=files(stage),
                    asset_manifest_sha256=digest(root / 'assets/manifest.json')
                    if (root / 'assets/manifest.json').exists() else None)
    write(stage / 'manifest.json', manifest)
    stage.rename(destination)
    event(root, 'checkpoint', checkpoint=name, note=note)
    return manifest


def restore_checkpoint(root, name):
    cp = root / 'checkpoints' / identifier(name)
    manifest = read(cp / 'manifest.json')
    actual = files(cp)
    actual.pop('manifest.json')
    if actual != manifest['files']:
        raise ValueError('Checkpoint changed')
    asset = root / 'assets/manifest.json'
    if manifest['asset_manifest_sha256'] and (not asset.exists() or digest(asset) != manifest['asset_manifest_sha256']):
        raise ValueError('Checkpoint belongs to different model/data assets')
    # Save the present state rather than discarding an unsubmitted edit.
    previous = checkpoint(root, note='Automatically saved before restoring ' + name)
    current = root / 'work/edited'
    current.rename(root / 'checkpoints' / ('.previous-edits-' + uuid.uuid4().hex))
    shutil.copytree(cp / 'edited', current)
    if (cp / 'notes.md').exists():
        shutil.copyfile(cp / 'notes.md', root / 'work/notes.md')
    else:
        (root / 'work/notes.md').unlink(missing_ok=True)
    memory = ensure(root)
    memory['next_step'] = manifest.get('next_step', manifest['note'])
    write(root / 'experiment.json', memory)
    # Restoring weights never rewinds forward or Python budgets.
    event(root, 'restore_checkpoint', checkpoint=name, previous=previous['id'])
    return manifest


def import_evidence(root, source):
    """Import a prior practice pilot as context, never as a free new budget."""
    source = Path(source).resolve()
    request = read(source / 'request.json')
    if request.get('benchmark') or request.get('split') in ('heldout', 'test') or request.get('type') not in ('generate', 'teacher_forced'):
        raise ValueError('Only practice evidence can enter editor context')
    names = ['request.json', 'plan.json', 'report.json', 'verified_report.json',
             'summary.json', 'pilot.json', 'prospective_metrics.json', 'verification.json']
    selected = {name: digest(source / name) for name in names if (source / name).is_file()}
    if not any(name in selected for name in ('summary.json', 'pilot.json')):
        raise ValueError('Require summary.json or pilot.json from a completed pilot')
    key = 'evidence-' + hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()[:16]
    target = root / 'evidence' / key
    if target.exists():
        if read(target / 'manifest.json')['files'] != selected:
            raise ValueError('Evidence identity conflict')
        return key
    stage = root / 'evidence' / ('.tmp-' + uuid.uuid4().hex)
    stage.mkdir()
    for name in selected:
        if (source / name).is_symlink():
            raise ValueError('Evidence symlinks are not supported')
        shutil.copyfile(source / name, stage / name)
    write(stage / 'manifest.json', dict(id=key, imported=time.time(), source=str(source),
          files=selected, accounting='external pilot; not a forward pass in this run'))
    stage.rename(target)
    event(root, 'import_evidence', evidence=key)
    return key


def context(root):
    state = read(root / 'state.json')
    memory = ensure(root)
    hypotheses = [read(p) for p in sorted((root / 'hypotheses').glob('*.json'))]
    jobs = []
    for job in sorted((root / 'jobs').glob('forward-*')):
        item = dict(id=job.name, intent=read(job / 'intent.json') if (job / 'intent.json').exists() else {},
                    request=read(job / 'request.json'))
        if seal(job):
            item['scores'] = read(job / 'output/scores.json')
        jobs.append(item)
    evidence = []
    for directory in sorted((root / 'evidence').glob('evidence-*')):
        manifest = read(directory / 'manifest.json')
        for name, expected in manifest['files'].items():
            if digest(directory / name) != expected:
                raise ValueError('Imported evidence changed')
        item = dict(id=directory.name, accounting=manifest['accounting'])
        for name in ('plan.json', 'report.json', 'verified_report.json', 'summary.json', 'prospective_metrics.json'):
            if (directory / name).exists():
                item[name] = read(directory / name)
        evidence.append(item)
    notes = root / 'work/notes.md'
    return dict(experiment=memory, budget=dict(total=state['budget'], used=state['used'],
          python_runs=state['python_runs'], frozen=state['frozen']), hypotheses=hypotheses,
          practice_jobs=jobs, imported_evidence=evidence,
          agent_actions=[dict(id=p.parent.name, **read(p)) for p in sorted((root / 'agent').glob('action-*/state.json'))][-12:],
          python_actions=[dict(id=p.name.removesuffix('.intent.json'), intent=read(p),
               code=p.with_name(p.name.removesuffix('.intent.json') + '.py').read_text(),
               result=read(p.with_name(p.name.removesuffix('.intent.json') + '.json'))
               if p.with_name(p.name.removesuffix('.intent.json') + '.json').exists() else None)
               for p in sorted((root / 'code').glob('*.intent.json'), key=lambda p:p.stat().st_mtime)[-8:]],
          notes=notes.read_text() if notes.exists() and not notes.is_symlink() else '',
          edits=files(root / 'work/edited'),
          checkpoints=[read(p) for p in sorted((root / 'checkpoints').glob('*/manifest.json'))])


def handoff(root):
    data = context(root)
    write(root / 'handoff.json', data)
    budget = data['budget']
    lines = ['# Experiment handoff', '', f"Experiment: {data['experiment']['id']}",
             f"Editor: gpt-6-astra / max. Phase: {data['experiment']['phase']}.",
             f"Forward passes: {budget['used']}/{budget['total']}; Python: {budget['python_runs']}/300.", '',
             '## Next step', '', data['experiment'].get('next_step') or 'Review hypotheses and practice results before selecting the next action.', '',
             '## Hypotheses', '']
    for h in data['hypotheses']:
        lines.extend([f"- {h['id']} [{h['status']}]: {h['statement']}", f"  Prediction: {h['prediction']}",
                      f"  Conclusion: {h['conclusion']}"])
    lines += ['', '## Practice jobs', '']
    for job in data['practice_jobs']:
        lines.append(f"- {job['id']} [{job['intent'].get('status', 'pending')}]: " + json.dumps(job.get('scores', {}), ensure_ascii=False))
    lines += ['', '## External practice evidence', '']
    for item in data['imported_evidence']:
        lines.extend([f"- {item['id']}: {item['accounting']}", json.dumps(item, ensure_ascii=False)])
    lines += ['', '## Notebook', '', data['notes'], '', '## Restart rules', '',
              'Kernel variables are not serialized. Reload weights with load()/load_original() and arrays from /work.',
              'Checkpoints restore edits and notes, never budgets. Completed requests must be reused by their job ID.',
              'Never read benchmark data or final evaluation results into the editor. Do not train or run a model in the CPU kernel.',
              'Only the trusted forward tool may execute the model. Use the same experiment root when continuing.', '']
    (root / 'handoff.md').write_text('\n'.join(lines))
    return data


def export_bundle(root, destination, include_weights=False):
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(root.resolve()):
        raise ValueError('Export to a new path outside the experiment root')
    entries = {}
    for p in sorted(root.rglob('*')):
        rel = p.relative_to(root)
        if p.is_symlink():
            raise ValueError(f'Cannot export symlink: {p}')
        if not p.is_file() or p.name in ('.lock', '.agent.lock') or '.cache' in rel.parts or any(x.startswith('.tmp-') for x in rel.parts):
            continue
        if not include_weights and rel.parts[:2] == ('assets', 'model') and p.suffix == '.safetensors':
            continue
        entries[str(rel)] = digest(p)
    temporary = destination.with_name(destination.name + '.tmp-' + uuid.uuid4().hex)
    with tarfile.open(temporary, 'w:gz') as archive:
        for rel in entries:
            # Add regular files explicitly; never follow directory recursion or links.
            info = archive.gettarinfo(str(root / rel), arcname=rel)
            with (root / rel).open('rb') as f:
                archive.addfile(info, f)
        import io
        payload = json.dumps(dict(version=1, files=entries, includes_weights=include_weights)).encode()
        info = tarfile.TarInfo('_bundle.json')
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    temporary.replace(destination)
    return len(entries)


def restore_bundle(source, target):
    target = Path(target).resolve()
    if target.exists():
        raise ValueError('Restore requires a new, nonexistent --root')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.wsp-restore-', dir=target.parent) as temporary:
        stage = Path(temporary) / 'run'
        stage.mkdir()
        with tarfile.open(source) as archive:
            names = set()
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or '\\' in member.name or not member.isfile() or str(path) != member.name or str(path) in names:
                    raise ValueError('Unsafe or duplicate archive member')
                names.add(member.name)
            archive.extractall(stage, filter='data')
        bundle = read(stage / '_bundle.json')
        actual = files(stage)
        actual.pop('_bundle.json')
        if actual != bundle['files'] or 'state.json' not in actual:
            raise ValueError('Bundle checksum mismatch')
        (stage / '_bundle.json').unlink()
        state = read(stage / 'state.json')
        state['container'] = 'wsp-' + uuid.uuid4().hex[:12]
        write(stage / 'state.json', state)
        memory = ensure(stage)
        memory['phase'] = 'paused'
        write(stage / 'experiment.json', memory)
        # Recreate directories omitted by a file-only archive.
        for directory in ('assets', 'work/edited', 'jobs', 'code'):
            (stage / directory).mkdir(parents=True, exist_ok=True)
        stage.rename(target)
    event(target, 'restore_bundle', source=str(Path(source).resolve()))
