'''
File: dataset_librispeech.py
Brief: Loaders for librispeech transcripts and audio.
'''

import os
import pandas as pd
import pathlib
from dataset import Dataset

WINDOW_LEN = .02 # sec

class LibriSpeechDataset(Dataset):

    @classmethod
    def load_transcripts(cls, filepath:pathlib.Path, audio_type='.flac', window_len=WINDOW_LEN):
        """
        This function is to get audios and transcripts needed for training
        @filepath: the path of the dicteory
        """
        count, k, inp = 0, 0, []
        audio_names, audio_trans = [], []
        for dir1 in filepath.iterdir():
            if not dir1.is_dir():
                continue
            for dir2 in dir1.iterdir():
                if not dir2.is_dir():
                    continue
                for audio in dir2.iterdir():
                    if audio.name.endswith('.txt'):
                        k += 1
                        with open(audio) as f:
                            line = f.readlines()
                            for item in line:
                                flac_path = os.path.join(audio.parent, item.split()[0] + audio_type)
                                audio_names.append(flac_path)
                                text = item.split()[1:]
                                text = ' '.join(text)
                                audio_trans.append(text)
        return pd.DataFrame({"path": audio_names, "transcripts": audio_trans})
    
