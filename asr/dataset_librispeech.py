'''
File: dataset_librispeech.py
Brief: Loaders for librispeech transcripts and audio.
'''

import os
import pandas as pd
from dataset import Dataset

WINDOW_LEN = .02 # sec

class LibriSpeechDataset(Dataset):

    @classmethod
    def load_transcripts(cls, filepath, audio_type='.flac', window_len=WINDOW_LEN):
        """
        This function is to get audios and transcripts needed for training
        @filepath: the path of the dicteory
        """
        count, k, inp = 0, 0, []
        audio_name, audio_trans = [], []
        for dir1 in os.listdir(filepath):
            if dir1 == '.DS_Store': continue
            dir2_path = os.path.join(filepath, dir1, os.sep)
            for dir2 in os.listdir(dir2_path):
                if dir2 == '.DS_Store': continue
                dir3_path = os.path.join(dir2_path, dir2, os.sep)
    
                for audio in os.listdir(dir3_path):
                    if audio.endswith('.txt'):
                        k += 1
                        trans_path = os.path.join(dir3_path, audio)
                        with open(trans_path) as f:
                            line = f.readlines()
                            for item in line:
                                flac_path = os.path.join(dir3_path, item.split()[0], audio_type)
                                audio_name.append(flac_path)
    
                                text = item.split()[1:]
                                text = ' '.join(text)
                                audio_trans.append(text)
        return pd.DataFrame({"path": audio_name, "transcripts": audio_trans})
    
