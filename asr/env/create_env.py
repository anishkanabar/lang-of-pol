import os
import subprocess
import argparse
import shutil
import sys
import pathlib

def main(env_dir:pathlib.Path):
    if os.path.exists(env_dir):
        print("Using existing env.")
        return 0
    
    if ' ' in str(env_dir):
        print(f"Can't use environment path with spaces:\n{env_dir}")
        return 1
    
    env_parent = os.path.dirname(env_dir)
    if not os.path.exists(env_parent):
        print("Environment directory not found. Create?")
        print(f"{env_dir}")
        needCreate = None
        while needCreate not in ["yes","no"]:
            needCreate = input("Type yes or no:\n")
        if needCreate == "yes":
            os.makedirs(env_parent)
        else:
            print("Terminating without creating environment.")
            return 1
    
    print("Creating conda environment.")
    cmd = f'conda create -y -c conda-forge -p "{env_dir}"'
    subprocess.run(cmd.split(), check=True)
    
    readme_path = os.path.join(env_parent, "README.txt")
    if not os.path.exists(readme_path):
        shutil.copy("ENV_README.txt", readme_path) 
    
    reqs_path = os.path.join(env_parent, "requirements.txt")
    if os.path.exists("requirements.txt"):
        if not os.path.exists(reqs_path):
            shutil.copy("requirements.txt", reqs_path)
        cmd = f'conda install -y -c conda-forge -p "{env_dir}" --file requirements.txt'
        subprocess.run(cmd.split())
    else:
        cmd = 'conda install -y -p "{env_dir}" "tensorflow-gpu==2.4.1" keras-gpu pandas "numpy==1.19.2"'
        subprocess.run(cmd.split())
        cmd = 'conda install -y -c conda-forge -p "{env_dir}" librosa'
        subprocess.run(cmd.split())

