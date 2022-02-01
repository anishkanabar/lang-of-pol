'''
File: dataset_librispeech.py
Brief: Loaders for librispeech transcripts and audio.
'''

import os
import pandas as pd
from asr_dataset.base_dataset import ASRDataset
from asr_dataset.constants import DATASET_DIRS

class LibriSpeechDataset(ASRDataset):
    
    def __init__(self, 
                 cluster: str='rcc', 
                 nrow: int=None, 
                 frac: float=None,
                 nsecs: float=None):
        self.dataset_path = DATASET_DIRS[cluster]['librispeech']
        super().__init__('librispeech', nrow, frac, nsecs)

    @classmethod
    def filter_manifest(cls, data: pd.DataFrame) -> pd.DataFrame:
        """ Don't need to filter this dataset. Data quality is good. """
        return data

    def create_manifest(self) -> pd.DataFrame:
        """
        Collect info on audio files and transcripts.
        Returns: DataFrame with columns:
            path - full path to audio file
            transcripts - transcript text
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
                                flac_path = os.path.join(dir3_path, item.split()[0]) + '.flac' 
                                audio_name.append(flac_path)
    
                                text = item.split()[1:]
                                text = ' '.join(text)
                                audio_trans.append(text)
        data = pd.DataFrame({"path": audio_name, "transcripts": audio_trans})
        return data
