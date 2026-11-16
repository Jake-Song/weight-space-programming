"""Summarize downloaded evaluations, with paired bootstrap uncertainty."""
import json, pathlib, random

root=pathlib.Path(__file__).resolve().parents[1]/'results'
summaries=[]
for p in sorted(root.glob('results/*/scores.json')):
    s=json.loads(p.read_text())
    summaries.append(s)
    print(f"{s['split']:8s} {s['condition']:8s} n={s['n']:3d} accuracy={s['accuracy']:.1%} cut={s['cut_share']:.1%} loop_proxy={s['loop_proxy_share']:.1%} tokens={s['mean_tokens']:.0f}")
paired=[]
interaction=None
direct=[]
base_path=root/'results/heldout_base_32768/per_problem.json'
if base_path.exists():
    base={r['unique_id']:r for r in json.loads(base_path.read_text())}
    for p in sorted(root.glob('results/heldout_*/per_problem.json')):
        rows=json.loads(p.read_text())
        assert set(base)=={r['unique_id'] for r in rows}
        differences=[int(r['correct'])-int(base[r['unique_id']]['correct']) for r in rows]
        rng=random.Random(0)
        means=sorted(sum(rng.choices(differences,k=len(rows)))/len(rows) for _ in range(10000))
        paired.append({'condition':p.parent.name,'accuracy_difference':sum(differences)/len(rows),
                       'paired_bootstrap_95_interval':[means[250],means[9749]]})
    print(json.dumps(paired,indent=2))
    paths={c:root/f'results/heldout_{c}_32768/per_problem.json' for c in ['layer15','layer19','both']}
    if all(p.exists() for p in paths.values()):
        arms={c:{r['unique_id']:r for r in json.loads(p.read_text())} for c,p in paths.items()}
        values=[int(arms['both'][i]['correct'])-int(arms['layer15'][i]['correct'])-int(arms['layer19'][i]['correct'])+int(b['correct']) for i,b in base.items()]
        rng=random.Random(0)
        boot=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(10000))
        interaction={'contrast':'both - layer15 - layer19 + base (accuracy scale)',
                     'estimate':sum(values)/len(values),'paired_bootstrap_95_interval':[boot[250],boot[9749]],
                     'note':'Exploratory, single generation per problem; not corrected for multiple comparisons.'}
        print(json.dumps(interaction,indent=2))
        for extra in ['nothink','random']:
            other_path=root/f'results/heldout_{extra}_32768/per_problem.json'
            if other_path.exists(): arms[extra]={r['unique_id']:r for r in json.loads(other_path.read_text())}
        for other in [c for c in ['layer15','layer19','nothink','random'] if c in arms]:
            d=[int(arms['both'][i]['correct'])-int(arms[other][i]['correct']) for i in base]
            rng=random.Random(0)
            boot=sorted(sum(rng.choices(d,k=len(d)))/len(d) for _ in range(10000))
            direct.append({'contrast':f'both - {other}','estimate':sum(d)/len(d),
                           'wrong_to_right':d.count(1),'right_to_wrong':d.count(-1),
                           'paired_bootstrap_95_interval':[boot[250],boot[9749]]})
        print(json.dumps(direct,indent=2))
(root/'summary.json').write_text(json.dumps({'evaluations':summaries,'paired_comparisons':paired,'interaction':interaction,'direct_comparisons':direct},indent=2))
