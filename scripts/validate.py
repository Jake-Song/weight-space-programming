import json, pathlib, sys, subprocess
root=pathlib.Path('/content/weight-space')
sys.path.insert(0,str(root/'vendor/code'))
import mathqa as Q
assert Q.is_correct(r'\frac{1}{2}','0.5')
assert not Q.is_correct('3','4')
assert not Q.is_correct(None,'4')
smoke=json.loads((root/'smoke.json').read_text())
heldout=json.loads((root/'heldout.json').read_text())
assert len(smoke)==20 and len(heldout)==100
assert not {r['unique_id'] for r in smoke}&{r['unique_id'] for r in heldout}
print('PASS: symbolic grading, missing-answer grading, and disjoint 20/100 split')
print(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv'],text=True))
