'''
File: atczero.py
Brief: Loader for ATC0 corpus.
Authors: William Dolan <wdolan@uchicago.edu>
'''

WINDOW_LEN = .02 #Sec
SAMPLE_RATE = 8000

import datetime
import pandas as pd
import logging
from asr_dataset.dataset import AudioClipDataset
from asr_dataset.datasets.constants import DATASET_DIRS

class ATCZeroDataset(AudioClipDataset):

    def __init__(self, cluster:str='rcc', nrow: int=None, frac: float=None, window_len=WINDOW_LEN):
        """
        Returns a ATCZeroDataset with a data attribute which is a dataframe of:
            path to utterance audio, duration (sec), number of samples, transcript text, [other columns]
        """
        self.transcripts_dir = DATASET_DIRS[cluster]['atc0']
        
        super().__init__('atczero', nrow, frac, window_len)
        #self.data = self.add_duration(self.data)

        self.data = self.add_sample_count(self.data)

    def check_number(x):
        return x.replace('.', '', 1).isdigit()

    def _load_transcripts(self, sample_rate=SAMPLE_RATE, window_len = WINDOW_LEN):
        """
        This function is to get audios and transcripts needed for training
        Returns: a dataframe with columns: path to utterance audio, transcripts
        """
        # XXX: This function is called by the base class's init method. It's kind of a weird 
        #       way to implement subclassing & maybe should be refactored. But for for now
        #       this function should do most of the class-specific data loading work.
        # XXX: This funciton is kind of weird. It is actually called by the base class's init method.
        #       So this function should do most of the work of

        unloaded_df = pd.read_csv(self.transcripts_dir + '/atc0.csv')

        loaded_df = unloaded_df

        loaded_df = loaded_df[loaded_df['start'].apply(lambda x: x.replace('.', '', 1).isdigit())]
        loaded_df = loaded_df[loaded_df['end'].apply(lambda x: x.replace('.', '', 1).isdigit())]

        ##TODO: Move to different func to fit formatting of radio.py?
        loaded_df['duration'] = loaded_df['end'].astype(float) - loaded_df['start'].astype(float)

        loaded_df['offset'] = loaded_df['start'].astype(float)

        loaded_df['path'] = loaded_df['filePath'].str.replace('atc0', self.transcripts_dir + '/atc0')
        loaded_df.drop(['start', 'end', 'filePath'], axis = 1, inplace = True)

        loaded_df = self._add_clip_paths(loaded_df)

        return loaded_df


    def _add_clip_paths(self, data):
        """
        Add column with path to audio clip.
        Params:
            @data: expects columns {path, offset, duration, transcripts}
        """
        msPerSec = 1000

        off_fmt = (data['offset'].astype(float) * msPerSec).astype('str')
        end_fmt = ((data['offset'].astype(float) + data['duration'].astype(float)) * msPerSec).astype('str')
        ext_fmt = pd.Series(['.sph']*len(data))
        clip_names = off_fmt.str.cat(end_fmt, '_').str.cat(ext_fmt)
        clip_paths = data['path'].str.replace('audio','audio/utterances', regex=False) \
                    .str.replace('.sph', '').str.cat(clip_names, '/')
        return data.assign(clip_path=clip_paths)