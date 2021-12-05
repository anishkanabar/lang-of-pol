import os
import argparse
import pathlib
import sys
from env.create_env import main as create_env

parser = argparse.ArgumentParser(description='Train asr model.')
parser.add_argument('env_dir', type=pathlib.Path,
                    help='path to virtual python environment for this model')
parser.add_argument('--local', type=bool, default=False,
                    help='run locally (otherwise on slurm')
args = parser.parse_args()

rtn = create_env(args.env_dir)

if rtn != 0:
    print("Failed to create conda env.")
    sys.exit(1)
elif args.local:
    subprocess.run(f'sh run-training.job "{args.env_dir}"')
else:
    subprocess.run(f'sbatch run-training.job "{args.env_dir}"')

