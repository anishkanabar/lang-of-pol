'''
File: dataset_librispeech.py
Brief: Loaders for librispeech transcripts and audio.
'''

import os
import pandas as pd
from dataset import Dataset
from dataset_locations import DATASET_DIRS

WINDOW_LEN = .02 # Sec

class LibriSpeechDataset(Dataset):
    
    def __init__(self, cluster: str='rcc', nrow: int=None, frac: float=None, window_len=WINDOW_LEN):
        self.dataset_path = DATASET_DIRS[cluster]['librispeech']
        super().__init__('librispeech', nrow, frac, window_len)

    def _load_transcripts(self, audio_type='.flac', window_len=WINDOW_LEN):
        """
        This function is to get audios and transcripts needed for training
        """
        count, k, inp = 0, 0, []
        audio_name, audio_trans = [], []
        for dir1 in os.listdir(self.dataset_path):
            if dir1 == '.DS_Store': continue
            dir2_path = os.path.join(self.dataset_path, dir1)
            for dir2 in os.listdir(dir2_path):
                if dir2 == '.DS_Store': continue
                dir3_path = os.path.join(dir2_path, dir2)
                for audio in os.listdir(dir3_path):
                    if audio.endswith('.txt'):
                        k += 1
                        trans_path = os.path.join(dir3_path, audio)
                        with open(trans_path) as f:
                            line = f.readlines()
                            for item in line:
                                flac_path = os.path.join(dir3_path, item.split()[0]) + audio_type
                                audio_name.append(flac_path)
    
                                text = item.split()[1:]
                                text = ' '.join(text)
                                audio_trans.append(text)
        return pd.DataFrame({"path": audio_name, "transcripts": audio_trans})
    
