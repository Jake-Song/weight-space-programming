import subprocess, sys
subprocess.run(['nvidia-smi'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'vllm==0.33.0', 'transformers==5.20.2', 'math-verify==0.9.0', 'datasets'], check=True)
print('SETUP_COMPLETE', flush=True)
