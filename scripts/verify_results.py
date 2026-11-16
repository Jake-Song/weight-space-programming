"""Check saved artifacts independently of summary construction."""
import json, pathlib
root=pathlib.Path(__file__).resolve().parents[1]/'results'
smoke=json.loads((root/'smoke.json').read_text())
heldout=json.loads((root/'heldout.json').read_text())
assert not {r['unique_id'] for r in smoke}&{r['unique_id'] for r in heldout}
checked=[]
for condition in ['base','layer15','layer19','both','nothink','random']:
    p=root/f'results/heldout_{condition}_32768'
    s=json.loads((p/'scores.json').read_text())
    rows=json.loads((p/'per_problem.json').read_text())
    raw=[json.loads(l) for l in (p/'generations.jsonl').read_text().splitlines()]
    assert len(rows)==len(raw)==s['n']==100
    assert [r['unique_id'] for r in rows]==[r['unique_id'] for r in raw]==[r['unique_id'] for r in heldout]
    assert sum(r['correct'] for r in rows)==s['correct']
    assert sum(r['correct'] for r in rows)/100==s['accuracy']
    assert sum(r['tokens'] for r in raw)/100==s['mean_tokens']
    assert sum(r['finish_reason']=='length' for r in raw)/100==s['cut_share']
    for r in raw:
        assert len(r['token_ids'])==r['tokens']<=32768
        assert r['finish_reason'] in ['length','stop']
        if r['finish_reason']=='length': assert r['tokens']==32768
    checked.append(condition)
audit={'verified_conditions':checked,'problems_per_condition':100,
       'checks':['disjoint pilot/heldout','same ordered problems','raw token lengths','termination reasons','aggregate scores']}
(root/'verification.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
