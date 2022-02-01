'''
File: base_dataset.py
Brief: Abstract base class for ASR dataset loaders
Authors: Eric Chandler <echandler@uchicago.edu>
'''

import abc
import pandas as pd
import logging
import librosa

logger = logging.getLogger('asr.dataset')

class ASRDataset(abc.ABC):

    def __init__(self, 
                 name: str, 
                 nrow: int=None, 
                 frac: float=None, 
                 nsecs: float=None):
        data = self.create_manifest()
        data = self._add_duration(data)
        data = self._sample(data, nrow, frac, nsecs)
        # Filtering can be expensive, so first we take subset (sample).
        # data = self._add_sample_count(data)
        data = self.filter_manifest(data)
        self.data = data
        self.name = name
        self.describe()

    @abc.abstractmethod
    def create_manifest(self) -> pd.DataFrame:
        """
        Collect info on audio files and transcripts.
        Returns: DataFrame with columns:
            path - full path to audio file
            transcripts - transcript text
            duration - length of audio in (float) seconds
        """
        pass

    @classmethod
    def filter_manifest(cls, data: pd.DataFrame) -> pd.DataFrame:
        """ Usually don't have to filter anything because data quailty is good."""
        return data

    @classmethod
    def _sample(cls, 
                data: pd.DataFrame, 
                nrow: int=None, 
                frac: float=None, 
                nsecs: float=None) -> pd.DataFrame:
        """
        Subset the data. Choose only one unit of size.
        Params:
            data: manifest. expects columns {duration}
            nrow: number of rows to keep
            frac: fraction of rows to keep
            nsecs: total duration to keep (in seconds)
        """ 
        if nsecs is not None:
            idx = data['duration'].cumsum().searchsorted(nsecs)
            return data.head(idx)
        elif nrow is not None:
            return data.head(nrow)
        elif frac is not None:
            return data.sample(frac=frac, random_state=1234)
        else:
            return data
        
    def describe(self):
        """
        Prints helpful statistics about dataset.
        """
        logger.info(f"{self.name} dataset stats:")
        logger.info(f"\tRow count = {self.data.shape[0]}")
        logger.info(f"\tMin duration = {self.data['duration'].min():.2f} (sec)")
        logger.info(f"\tMax duration = {self.data['duration'].max():.2f} (sec)")
        logger.info(f"\tMean duration = {self.data['duration'].mean():.2f} (sec)")
        logger.info(f"\tStdev duration = {self.data['duration'].std():.2f} (sec)") 
        logger.info(f"\tTotal duration = {pd.Timedelta(self.data['duration'].sum(),'sec')}") 
    
    @classmethod
    def _add_duration(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        Params:
            data: expects columns {path}
        Returns: 
            dataframe with columns {original columns..., duration}
        """
        if 'duration' in data.columns:
            return data
        else:
            duration_func = lambda x: librosa.get_duration(filename=x)
            return data.assign(duration=data['path'].apply(duration_func))

    @classmethod
    def _add_sample_count(cls, data: pd.DataFrame):
        """
        Params:
            data: expects columns {path, duration}
        Returns: 
            dataframe with columns {original columns..., nsamples}
        """
        if 'nsamples' in data.columns:
            return data
        else:
            data_ = data.assign(sr=data['path'].apply(librosa.get_samplerate))
            count_func = lambda row: librosa.time_to_samples(row['duration'], row['sr'])
            sample_count = data_.apply(lambda row: count_func(row), axis=1)
            return data.assign(nsamples=sample_count)

