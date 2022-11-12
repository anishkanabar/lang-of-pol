""" Minimal reproducible test that torchaudio is working on cluster."""

import torch
import torchaudio

filename = '/project/graziul/data/utterances/Zone1/2018_08_11/201808112007-612624-27730/1881828_1882249.flac'

try:
    x = torch.ops.torchaudio.sox_io_load_audio_file(filename)
    print('Successfully loaded audio via sox.')
except RuntimeError as e:
    print('Failed loading audio with sox.')
    print(e)

try:
    x, _ = torchaudio.load(filename)
    print('Successfully loaded audio via torchaudio.')
except RuntimeError as e:
    print('Failed loading audio with torchaudio.')
    print(e)
