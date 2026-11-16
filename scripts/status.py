import pathlib, subprocess, json, time
root=pathlib.Path('/content/weight-space')
print((root/'status.json').read_text() if (root/'status.json').exists() else 'No status yet')
status=json.loads((root/'status.json').read_text())
if 'started' in status: print('Stage elapsed seconds:',round(time.time()-status['started']))
f=root/(status.get('stage','suite')+'.log')
if f.exists(): print(f.name, f.read_text()[-1200:])
for f in sorted(root.glob('results/*/scores.json')):
    import json
    s=json.loads(f.read_text())
    print(f.parent.name,{k:s[k] for k in ['n','accuracy','cut_share','loop_proxy_share','mean_tokens','elapsed_seconds']})
