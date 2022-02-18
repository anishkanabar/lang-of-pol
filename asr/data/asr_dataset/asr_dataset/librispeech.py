'''
File: dataset_librispeech.py
Brief: Loaders for librispeech transcripts and audio.
'''

import os
import pandas as pd
from asr_dataset.base import AsrETL
from asr_dataset.constants import DATASET_DIRS, Cluster


class LibriSpeechETL(AsrETL):
    
    def __init__(self, cluster: Cluster=Cluster.RCC):
        super().__init__('librispeech')
        self.dataset_path = DATASET_DIRS[cluster]['librispeech_data']


    def extract(self) -> pd.DataFrame:
        """
        Collect info on audio files and transcripts in their original form.

        Returns: DataFrame with columns:
            audio - (str) full path to audio file
            text - (str) transcript text
            duration - (float) length of audio in seconds
        """
        data = self._create_manifest()
        data = self._add_duration(data)
        self.describe(data)
        return data


    def transform(self, data: pd.DataFrame, sample_rate: int=None) -> pd.DataFrame:
        """ Dataset already clean and in useful form """
        return data
        

    def load(self, 
             data: pd.DataFrame=None, 
             qty:Real=None, 
             units: DataSizeUnit=None) -> pd.DataFrame:
        """
        Collect info on the transformed audio files and transcripts.
        Does NOT load waveforms into memory.

        Returns: DataFrame with same columns as extract()
        """
        if data is None:
            data = self.extract()
        data = self._sample(data, qty, units)
        self.describe(data)
        return data
        

    def _create_manifest(self) -> pd.DataFrame:
        """
        Collect info on audio files and transcripts.
        Returns: DataFrame with columns:
            audio - full path to audio file
            text - transcript text
        """
        audios, texts = [], []
        for root, dirs, files in os.walk(self.dataset_path):
            manifest_names = [x for x in files if x.endswith('.txt')]
            for manifest_name in manifest_names:
                manifest_file = os.path.join(root, manifest_name)
                with open(manifest_file) as f:
                    line = f.readlines()
                    for item in line:
                        audio_name = item.split()[0]
                        audio_text = ' '.join(item.split()[1:])
                        audios.append(audio_name)
                        texts.append(audio_text)
        data = pd.DataFrame({"audio": audios, "text": texts})
        return data

