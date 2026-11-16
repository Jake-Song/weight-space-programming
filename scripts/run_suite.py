import json, pathlib, subprocess, sys, time, traceback

root=pathlib.Path('/content/weight-space')
root.mkdir(exist_ok=True)

def run(cmd,label):
    (root/'status.json').write_text(json.dumps({'stage':label,'started':time.time(),'status':'running'}))
    with (root/f'{label}.log').open('w') as f:
        subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)

try:
    run([sys.executable,'/content/prepare_wsp.py'],'prepare')
    for c in ['base','both']:
        run([sys.executable,'/content/evaluate_wsp.py','--condition',c,'--split','smoke'],f'smoke_{c}')
    scores={c:json.loads((root/'results'/f'smoke_{c}_32768'/'scores.json').read_text()) for c in ['base','both']}
    # Predeclared pilot gate: continue when there is either accuracy or cut-rate improvement.
    proceed=(scores['both']['accuracy']>scores['base']['accuracy'] or
             scores['both']['cut_share']<scores['base']['cut_share'])
    if proceed:
        for c in ['base','layer15','layer19','both','nothink']:
            run([sys.executable,'/content/evaluate_wsp.py','--condition',c,'--split','heldout'],f'heldout_{c}')
    (root/'status.json').write_text(json.dumps({'status':'complete','heldout_run':proceed,'ended':time.time()}))
except Exception:
    (root/'status.json').write_text(json.dumps({'status':'failed','error':traceback.format_exc()}))
    raise
