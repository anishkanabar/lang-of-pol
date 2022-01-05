Files:
- requirements_deepspeech2_gpu/cpu.txt
    Python dependencies for running the deepspeech2 model on midway3, installed via conda.

Build:
To rebuild these requirements on another system, run the following. The version numbers are significant.
# GPU version
conda install -y "tensorflow-gpu==2.4.1" keras-gpu pandas "numpy==1.19.2"
# CPU version
conda install -y "tensorflow==2.4.1" keras pandas "numpy==1.19.2"
# Both versions
conda install -y -c conda-forge librosa

