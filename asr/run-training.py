import os
import argparse
import pathlib
import sys
import subprocess
from env.create_env import main as create_env

parser = argparse.ArgumentParser(description='Train asr model.')
parser.add_argument('env_dir', type=pathlib.Path,
                    help='path to virtual python environment for this model')
parser.add_argument('--local', dest='local', action='store_true',
                    help='run locally (otherwise on slurm')
args = parser.parse_args()

rtn = create_env(args.env_dir, args.local)

if rtn != 0:
    print("Failed to create conda env.")
    sys.exit(1)
elif args.local:
    cmd = f'sh run-training.job {args.env_dir}'
    subprocess.run(cmd.split())
else:
    cmd = f'sbatch run-training.job {args.env_dir} {args.local}'
    subprocess.run(cmd.split())

