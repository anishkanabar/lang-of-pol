# ASR Dataset
## A python library for creating ASR corpora manifests.

This module performs two tasks:
1) (As required) slices audio files into utterance-level files and saves them to disk.
2) Creates and returns the manifest file (an association of audio filepaths to transcript text).

This module does not load the audio into memory because most ASR models support lazy loading of audio files directed by a manifest file.
Note, what is useful for ASR may not be useful for VAD, SER, etc. In particular, we drop silence, our unit of analysis is an utterance,
and our audio frame-window thresholds are different.

## Installation
First activate your conda environment or venv.
> cd to this directory
> pip install -e .

## Usage

```
from asr_dataset.librispeech import LibriSpeechDataset
data = LibriSpeechDataset('rcc', nrow=3000).data
```

