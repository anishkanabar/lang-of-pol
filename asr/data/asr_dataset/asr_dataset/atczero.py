'''
File: atczero.py
Brief: Loader for ATC0 corpus.
Authors: William Dolan <wdolan@uchicago.edu>
'''

WINDOW_LEN = .02 #Sec
SAMPLE_RATE = 8000

import os
import datetime
import pandas as pd
import logging
from asr_dataset.utterance_dataset import UtteranceDataset
from asr_dataset.constants import DATASET_DIRS

class ATCZeroDataset(UtteranceDataset):

    def __init__(self, 
                cluster:str='rcc', 
                nrow: int=None, 
                frac: float=None, 
                nsecs: float=None):
        self.transcripts_dir = DATASET_DIRS[cluster]['atczero']
        super().__init__('atczero', nrow, frac, nsecs)


    def create_manifest(self) -> pd.DataFrame:
        """
        Collect info on audio files and transcripts.
        Returns: DataFrame with columns:
            path - full path to utterance audio file
            transcripts - transcript text
            duration - utterance audio length in seconds
            offset - utterance offset from beginning of context audio
            context_path - path to audio file with multiple utterances
            location - airport identifier
            speaker - air traffic controller id
            recipient - airplane id
            transcriber - transcriber id
        """
        unloaded_df = pd.read_csv(self.transcripts_dir + '/atc0.csv')

        loaded_df = unloaded_df

        loaded_df = loaded_df[loaded_df['start'].apply(lambda x: x.replace('.', '', 1).isdigit())]
        loaded_df = loaded_df[loaded_df['end'].apply(lambda x: x.replace('.', '', 1).isdigit())]

        ##TODO: Move to different func to fit formatting of radio.py?
        loaded_df['duration'] = loaded_df['end'].astype(float) - loaded_df['start'].astype(float)

        loaded_df['offset'] = loaded_df['start'].astype(float)

        loaded_df['context_path'] = self.transcripts_dir + os.sep + loaded_df['filePath']

        loaded_df = loaded_df.drop(['start', 'end', 'filePath'], axis = 1)

        loaded_df = self._add_utterance_paths(loaded_df)

        loaded_df = loaded_df.rename(columns={"transcription": "transcripts"})
        return loaded_df


    def _add_utterance_paths(self, data):
        """
        Add column with path to utterance audio clip.
        Params:
            @data: expects columns {context_path, offset, duration}
        """
        msPerSec = 1000

        off_fmt = (data['offset'].astype(float) * msPerSec).astype('str')
        end_fmt = ((data['offset'].astype(float) + data['duration'].astype(float)) * msPerSec).astype('str')
        ext_fmt = pd.Series(['.sph']*len(data))
        utterance_names = off_fmt.str.cat(end_fmt, '_').str.cat(ext_fmt)
        utterance_paths = data['context_path'].str.replace('audio','audio/utterances', regex=False) \
                    .str.replace('.sph', '').str.cat(utterance_names, '/')
        return data.assign(path=utterance_paths)
