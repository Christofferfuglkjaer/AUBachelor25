import sys
import subprocess
# implement pip as a subprocess:
dependencies = ['numpy', 'neptune', 'medmnist']
for dependency in dependencies:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', dependency])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4'])