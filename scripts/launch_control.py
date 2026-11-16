import subprocess, sys
with open('/content/weight-space/control_supervisor.log','w') as f:
    p=subprocess.Popen([sys.executable,'/content/run_wsp_control.py'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('CONTROL_SUPERVISOR_PID',p.pid)
