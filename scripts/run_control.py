import json, pathlib, subprocess, sys, time, traceback
root=pathlib.Path('/content/weight-space')
while True:
    try: status=json.loads((root/'status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError): status={}
    if status.get('status')=='failed': raise RuntimeError(status)
    if status.get('status')=='complete': break
    time.sleep(15)
if status.get('heldout_run') is False:
    print('Pilot gate not met; no held-out control requested.')
    sys.exit(0)
try:
    base=json.loads((root/'results/heldout_base_32768/scores.json').read_text())
    both=json.loads((root/'results/heldout_both_32768/scores.json').read_text())
    if both['accuracy']>base['accuracy'] or both['cut_share']<base['cut_share']:
        (root/'status.json').write_text(json.dumps({'status':'running','stage':'random_control'}))
        with (root/'random_control.log').open('w') as f:
            subprocess.run([sys.executable,'/content/random_wsp_control.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,'/content/evaluate_wsp.py','--condition','random','--split','heldout'],stdout=f,stderr=subprocess.STDOUT,check=True)
        (root/'status.json').write_text(json.dumps({'status':'complete','random_control':True,'ended':time.time()}))
except Exception:
    (root/'status.json').write_text(json.dumps({'status':'failed','error':traceback.format_exc()}))
    raise
