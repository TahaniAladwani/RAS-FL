"""Run tests in isolated processes to keep dataset-specific modules separate."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
for directory in ['core', 'baselines/fashionmnist', 'baselines/cifar10', 'baselines/cifar100', 'tests']:
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', directory, '-p', 'test_*.py', '-v'],
                   cwd=root, check=True)
