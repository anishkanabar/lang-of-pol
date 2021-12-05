import os
import subprocess
import argparse
import shutil
import sys
import pathlib

def main(env_dir:pathlib.Path, scale:str):
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
    cmd = f'conda create -y -c conda-forge -p {env_dir}'
    subprocess.run(cmd.split(), check=True)
    
    readme_src = os.path.join(os.path.dirname(__file__), "ENV_README.txt")
    readme_dest = os.path.join(env_parent, "README.txt")
    if not os.path.exists(readme_dest):
        shutil.copy(readme_src, readme_dest) 
    
    reqs_src = os.path.join(os.path.dirname(__file__), "requirements.txt")
    reqs_dest = os.path.join(env_parent, "requirements.txt")
    if os.path.exists(reqs_src) and scale == 'cluster':
        if not os.path.exists(reqs_dest):
            shutil.copy(reqs_src, reqs_dest)
        cmd = f'conda install -y -c conda-forge -p {env_dir} --file {reqs_dest}'
        subprocess.run(cmd.split(), check=True)
    else:
        tf_pkg = "tensorflow==2.4.1" if scale == 'local' else "tensorflow-gpu==2.4.1"
        np_pkg = "numpy==1.19.2" if scale == 'local' else "numpy==1.19.2"
        keras_pkg = "keras" if scale == 'local' else "keras_gpu"
        cmd = f'conda install -y -p {env_dir} "{tf_pkg}" {keras_pkg} pandas "{np_pkg}"'
        subprocess.run(cmd.split(), check=True)
        cmd = f'conda install -y -c conda-forge -p {env_dir} librosa'
        subprocess.run(cmd.split(), check=True)
    
    return 0

