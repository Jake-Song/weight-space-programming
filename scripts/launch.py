import pathlib, subprocess, sys
root=pathlib.Path('/content/weight-space')
root.mkdir(exist_ok=True)
with (root/'suite.log').open('w') as f:
    p=subprocess.Popen([sys.executable,'/content/run_wsp_suite.py'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('SUITE_PID',p.pid)
