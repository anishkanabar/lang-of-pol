# Speaker Diarization and Conversation Segmentation Pipeline

## Installation
### Librosa Dependency
Midway3 compute nodes don't have libsndfile (a common audio dependency),
so we have to install it and tell the compute nodes where to find it.
Please create a **separate** conda environment called "soundfile" exclusively for this purpose.
```
conda create -y -c conda-forge --name soundfile librosa
```
### Pyannote Dependencies
Try installing the requirements.txt into a conda environment.
If that doesn't work, follow instructions from [pyannote](https://github.com/pyannote/pyannote-audio) package.
