"""Paired English rendition of the Korean fact-update task, scored with thinking off."""
import argparse
from pathlib import Path
import json
import re

try:
    from . import fact_update_ko as fact, worker
except ImportError:
    import fact_update_ko as fact
    import worker

ENGLISH_FRAMES = [
    ('Current registered value', 'Requested new value', 'The request was approved.', 'The request was withdrawn.'),
    ('Original assignment', 'Proposed assignment', 'The proposed change was made.', 'The proposal was voted down.'),
    ('Previous entry', 'Proposed revision', 'The revision was incorporated.', 'The revision was discarded.'),
    ('Current entry', 'Proposed replacement', 'The decision was made to proceed with the replacement.',
     'The decision was made not to proceed with the replacement.'),
]
VALUES = {
    'room': ['Garam Room', 'Nuri Room', 'Dasom Room', 'Boram Room', 'Maru Room', 'Sodam Room',
             'Haesol Room', 'Hangyeol Room'],
    'schedule': ['Monday morning', 'Tuesday morning', 'Wednesday morning', 'Thursday morning',
                 'Friday morning', 'Monday afternoon', 'Tuesday afternoon', 'Wednesday afternoon'],
}
FIELDS = {'room': 'meeting room', 'schedule': 'meeting time'}


SELECTION_FRAMES = [
    ('Initially announced value', 'Requested correction', 'The correction notice was issued.', 'Publication of the correction notice was canceled.'),
    ('Initial designation', 'Replacement proposal', 'The replacement proposal passed.', 'The replacement proposal was rejected.'),
    ('Ledger entry', 'Value on the application', 'The ledger was amended according to the application.', 'The application was returned and the ledger left unchanged.'),
    ('Earlier decision', 'Motion for reconsideration', 'The motion will be implemented.', 'The earlier decision will stand.'),
]
FINAL_FRAMES = [
    ('Previous announcement', 'New motion', 'The new motion passed review.', 'The new motion failed review.'),
    ('Original register entry', 'Proposed change', 'The proposed change received final approval.', 'Approval of the proposed change was denied.'),
    ('Initial assignment', 'Follow-up request', 'The follow-up request was accepted.', 'The follow-up request was withdrawn.'),
    ('Existing instruction', 'Proposed replacement instruction', 'The replacement instruction was adopted.', 'The replacement instruction was rejected.'),
    ('First recorded value', 'Newly suggested value', '', ''),
    ('Value in the first record', 'Value in the revision request', '', ''),
    ('Existing specification', 'Replacement specification', '', ''),
    ('Earlier entry', 'New draft entry', '', ''),
]
IMPLICIT = {
    'delivery': [
        ('The waybill was reissued with the new address.', 'The waybill was issued with the original address.'),
        ('The package was sent to the site in the revision request.', 'The package was sent to the site in the first record.'),
        ('The driver delivered the goods to the store in the replacement specification.', 'The driver delivered the goods to the store in the existing specification.'),
        ('The receipt bears the stamp of the branch in the new draft entry.', 'The receipt bears the stamp of the branch in the earlier entry.')],
    'owner': [
        ('The newly suggested person took over the task.', 'The person first recorded continued handling the task.'),
        ('The work passed to the person in the revision request.', 'The work stayed with the person in the first record.'),
        ('The nameplate for the person in charge shows the name from the replacement specification.', 'The nameplate for the person in charge shows the name from the existing specification.'),
        ('The handover form lists the name from the new draft entry as the current owner.', 'The handover form lists the previously recorded name as the current owner.')],
    'archive': [
        ('The document was locked in the newly suggested box.', 'The document was locked in the box first recorded.'),
        ('The original was moved to the box in the revision request.', 'The original remained in the box in the first record.'),
        ('The archivist filed the original in the box in the replacement specification.', 'The archivist filed the original in the box in the existing specification.'),
        ('During inspection the original was found in the box in the new draft entry.', 'During inspection the original was found in the previously recorded box.')],
}
PRIVATE_VALUES = {
    'room': ['Ideun Room', 'Eunsol Room', 'Narae Room', 'Danbi Room', 'Yeoul Room', 'Dodam Room', 'Yunseul Room', 'Saebom Room'],
    'schedule': ["nine o'clock", "ten o'clock", "eleven o'clock", 'noon', 'one in the afternoon', 'two in the afternoon', 'three in the afternoon', 'four in the afternoon'],
    'delivery': ['Solbit Branch', 'Byeolha Branch', 'Damsol Branch', 'Goun Branch', 'Onsaem Branch', 'Pureum Branch', 'Yeonu Branch', 'Saetbyeol Branch'],
    'owner': ['Kim Suhyeon', 'Park Jimin', 'Oh Sejin', 'Yu Jeongmin', 'Choi Doyun', 'Lee Seohyeon', 'Lim Subin', 'Kang Yewon'],
    'archive': ['teal box', 'purple box', 'orange box', 'navy box', 'pink box', 'gray box', 'brown box', 'light green box'],
}
FIELDS.update(delivery='delivery destination', owner='task owner', archive='document storage location')


def control(row):
    domain = row['domain']
    i = int(row['id'].split('-')[-1])
    field, vocab = fact.DOMAINS[domain]
    mapping = dict(zip(vocab.split(), PRIVATE_VALUES[domain], strict=True))
    options = [mapping[line[3:]] for line in row['problem'].splitlines()[2:4]]
    answer = 'AB'.index(row['answer'])
    record = f'Record {i+1}. Case A lists the following {FIELDS[domain]}: {options[answer]}.'
    if row['kind'] == 'other_entity':
        record += f' Only case B changed. The new {FIELDS[domain]} for case B is {options[1-answer]}.'
    return dict(row, problem=f'{record}\nWhat is the currently valid {FIELDS[domain]} for case A?\nA. {options[0]}\nB. {options[1]}\nAnswer with just one letter, A or B.')


def prepare(assets):
    for split in ('explore', 'selection', 'final'):
        rows = [translate(row) for row in fact.dataset(split)]
        if split == 'final':
            rows.extend(control(row) for row in fact.preservation())
        path = assets / ('practice' if split == 'explore' else 'bench') / (split + '.jsonl')
        path.parent.mkdir(exist_ok=True)
        path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))


def translate(row):
    domain = row['domain']
    split = row['template'].split('-')[0]
    field, *vocabs = fact.DOMAINS[domain]
    vocab = vocabs[split == 'selection']
    english = VALUES if split == 'explore' else PRIVATE_VALUES
    values = dict(zip(vocab.replace('_', '~').split(), english[domain], strict=True))
    frame = int(row['template'].split('-')[-1])
    frames = {'explore': ENGLISH_FRAMES, 'selection': SELECTION_FRAMES, 'final': FINAL_FRAMES}
    labels = dict(zip(fact.FRAMES[split][frame], frames[split][frame], strict=True))
    if row['implicit']:
        labels.update(zip(fact.IMPLICIT[domain][frame-4], IMPLICIT[domain][frame-4], strict=True))
    header, first, second, outcome, question, a, b, instruction = row['problem'].splitlines()
    number = re.fullmatch(r'기록 (\d+) \(' + re.escape(field) + r'\)', header).group(1)
    if question != f'현재 유효한 {field} 항목의 값은 무엇입니까?' or instruction != '답으로 A 또는 B 한 글자만 쓰세요.':
        raise ValueError('Unexpected source question or instruction')
    lines = [f'Record {number} ({FIELDS[domain]})']
    for entry in (first, second):
        label, value = entry.split(': ')
        lines.append(f'{labels[label]}: {values[value.replace(" ", "~")]}')
    lines.extend([labels[outcome], f'What is the currently valid {FIELDS[domain]}?'])
    for letter, option in zip('AB', (a,b)):
        prefix = letter + '. '
        if not option.startswith(prefix):
            raise ValueError('Unexpected answer order')
        lines.append(prefix + values[option[len(prefix):].replace(' ', '~')])
    lines.append('Answer with just one letter, A or B.')
    return dict(row, problem='\n'.join(lines))


def run(root):
    from huggingface_hub import snapshot_download
    import importlib.metadata
    import torch
    source = root / 'explore-ko.jsonl'
    expected = json.loads((root/'ko-manifest.json').read_text())['files']
    if worker.digest(source) != expected['practice/explore.jsonl']:
        raise ValueError('Input differs from Korean run')
    worker.ASSETS = root
    worker.JOB = root
    practice = root / 'practice'
    practice.mkdir(exist_ok=True)
    output = root / 'output'
    output.mkdir(exist_ok=True)
    rows = [translate(row) for row in worker.jsonl(source)]
    path = practice / 'explore.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    ckpt = root / 'model'
    snapshot_download('Qwen/Qwen3.5-0.8B', revision=worker.REVISION, local_dir=ckpt,
                      allow_patterns=[key[6:] for key in expected if key.startswith('model/')])
    for key, digest in expected.items():
        if key.startswith('model/') and worker.digest(root/key) != digest:
            raise ValueError('Model asset mismatch: '+key)
    scores = worker.score_choices({'type':'score_choices'}, ckpt, 'original')
    worker.write(output/'scores.json', dict(scores=scores, model_revision=worker.REVISION,
                 english_sha256=worker.digest(path), korean_sha256=worker.digest(source),
                 thinking=False, gpu=torch.cuda.get_device_name(), worker_sha256=worker.digest(worker.__file__),
                 versions={p:importlib.metadata.version(p) for p in ('transformers','torch','safetensors')}))
    print(scores, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root', type=Path)
    run(cli.parse_args().root)
