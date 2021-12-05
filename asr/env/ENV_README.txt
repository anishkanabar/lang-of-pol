A pre-built conda environment for running Tensorflow + Keras on Midway 3. The requirements.txt is source controlled, but copied here for quick reference. This environment exists because midway3 computational nodes don't have direct internet access and therefore can't run `conda install ...` themselves.

Files:
This directory should contain: 
    - requirements.txt
        Recipe for tensorflow_env. Automatically copied here on creation for convenience.
    - <something>_env
        The conda environment.

Usage:
    If <something>_env is missing, run create_env.sh in the shell.

