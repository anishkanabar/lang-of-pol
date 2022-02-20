# Speaker Diarization and Conversation Segmentation Pipeline

## Installation
Create a conda environment called pyannote. It must be called pyannote. Install pytorch, torchaudio, **and librosa** to it , as well as the pyannote development library. (Without the explicit librosa call, conda will use a system-wide version of soundfile instead of downloading it into the environment. We need it in the environment for slurm to work -- see run.sh)
1. `conda create -n pyannote python=3.8`
2. `conda activate pyannote`
3. `conda install -c pytorch -c conda-forge pytorch torchaudio librosa`
4. `pip install https://github.com/pyannote/pyannote-audio/archive/develop.zip`

### PyTorch soundfile hack
Pyannote and PyTorch both mis-use the soundfile API and throw [errors](https://github.com/pytorch/audio/issues/2234). Our workaround is editing the downloaded pytorch library with a bugfix.
1. Open soundfile\_backend.py in your conda environment. e.g.
   `vim ~/.conda/envs/pyannote/lib/python3.8/site-packages/torchaudio/backend/soundfile_backend.py`
2. Edit line 102 in the info() function:
   `sinfo = soundfile.info(str(filepath))`

## Usage
Use the bash script "run.sh" to run a python script on the cluster.
e.g. `bash run.sh vad.py ami 1 1`

