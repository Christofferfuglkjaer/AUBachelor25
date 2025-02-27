import sys
import subprocess
# implement pip as a subprocess:
dependencies = ['numpy', 'matplotlib', 'tensorflow', 'keras', 'neptune', 'medmnist']
for dependency in dependencies:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', dependency])