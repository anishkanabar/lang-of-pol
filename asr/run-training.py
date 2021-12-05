import os
import argparse
import pathlib
import sys
import subprocess
from env.create_env import main as create_env

parser = argparse.ArgumentParser(description='Train asr model.')
parser.add_argument('scale', choices=['local','cluster'])
parser.add_argument('env_dir', type=pathlib.Path,
                    help='path to virtual python environment for this model')
args = parser.parse_args()

rtn = create_env(args.env_dir, args.scale)

if rtn != 0:
    print("Failed to create conda env.")
    sys.exit(1)
elif args.scale == 'local':
    logs_dir='/Users/eric/Documents/Work/PoliceBroadcasts/output_logs'
    dataset_dir='/Users/eric/Documents/Work/PoliceBroadcasts/FakeData'
    cmd = f'sh run-training.job local {args.env_dir} {dataset_dir} {logs_dir}'
    subprocess.run(cmd.split()) 
else:
    logs_dir='/project/graziul/ra/echandler'
    dataset_dir='/project/graziul/transcripts'
    cmd = f'sbatch run-training.job cluster {args.env_dir} {dataset_dir} {logs_dir}'
    subprocess.run(cmd.split())

