'''
File: base.py
Brief: Abstract base class for ASR dataset loaders
Authors: Eric Chandler <echandler@uchicago.edu>
'''

import os
import abc
import logging
import warnings
import datetime as dt
from numbers import Real
from enum import Enum, auto
import librosa
import soundfile
import pandas as pd
from asr_dataset.constants import DataSizeUnit


logger = logging.getLogger('asr.etl')


class AsrETL(abc.ABC):
    """ Base class for ASR dataset loaders """

    WINDOW_LEN = 0.04  # sec
    SAMPLE_RATE = 16000  # Hz

    def __init__(self, name: str):
        """
        Params:
            name: name of the dataset
        """
        self.name = name


    def etl(self, **kwargs) -> pd.DataFrame:
        """ Perform extract, transform, and load steps """
        data = self.extract()
        data = self.transform(data)
        return self.load(data, **kwargs)
    

    @abc.abstractmethod
    def extract(self) -> pd.DataFrame:
        """
        Collect info on audio files and transcripts in their original form.

        Returns: DataFrame with columns:
            audio - (str) full path to audio file
            text - (str) transcript text
            duration - (float) length of audio in seconds
        """
        pass
    

    @abc.abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Modify dataset into a form more useful for ASR. This may write to disk. 
        May also load audio files into memory.

        This function is intended to run ONCE over the lifetime of the dataset,
        but is a NO-OP if called again. To re-transform the dataset, delete the 
        output of this function.
        """
        pass


    @abc.abstractmethod
    def load(self, 
            data: pd.DataFrame=None, 
            qty:Real=None, 
            units: DataSizeUnit=None) -> pd.DataFrame:
        """
        Collect info on the transformed audio files and transcripts.

        Returns: DataFrame with same columns as extract()
        """
        pass


    def describe(self, data: pd.DataFrame, name_suffix: str=''):
        """
        Prints helpful statistics about dataset.
        """
        logger.info(f"{self.name}{name_suffix} dataset stats:")
        self.__class__._describe(data)


    @classmethod
    def _describe(cls, data: pd.DataFrame):
        logger.info(f"\tRow count = {data.shape[0]}")
        logger.info(f"\tMin duration = {data['duration'].min():.2f} (sec)")
        logger.info(f"\tMax duration = {data['duration'].max():.2f} (sec)")
        logger.info(f"\tMean duration = {data['duration'].mean():.2f} (sec)")
        logger.info(f"\tStdev duration = {data['duration'].std():.2f} (sec)")
        logger.info(f"\tTotal duration = {pd.Timedelta(data['duration'].sum(),'sec')}")


    ###################################
    ## Helper methods for Extract #####
    ###################################


    @classmethod
    def _add_duration(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds duration column to data frame, if necessary.
        Params:
            data: expects columns {audio}
        Returns:
            dataframe with columns {original columns..., duration}
        """
        if 'duration' in data.columns:
            return data
        else:
            # This is MUCH faster sequentially than via pd.apply
            # because you dont want to open so many file pointers concurrently
            durations = []
            for audio_path in data['audio']:
                durations.append(librosa.get_duration(filename=audio_path))
            return data.assign(duration=durations)
    

    ###################################
    ## Helper methods for Transform ###
    ###################################


    @classmethod
    def _audio_slicer(cls, offset: float, duration: float, sample_rate: int) -> slice:
        offset_idx = librosa.time_to_samples(offset, sr=sample_rate)
        duration_idx = librosa.time_to_samples(offset + duration, sr=sample_rate)
        return slice(offset_idx, duration_idx)


    def _write_utterances(self, data: pd.DataFrame, sample_rate: int=None):
        """
        Extract small audio clips from original file and write them to disk.
    
        This function is intended to run ONCE over the lifetime of the dataset,
        but is a NO-OP if called again. To re-transform the dataset, delete the
        output of this function.

        Params:
            data: expects columns {audio, original_audio, offset, duration}
            sample_rate: resample audio to this rate
        """
        logger.info('Writing utterance audio clips.')
        start = dt.datetime.now()

        original_audios = set(data['original_audio'])
        num_audios = len(original_audios)
        for idx, original_audio in enumerate(original_audios):
            utterances = data.loc[data['original_audio'] == original_audio]

            # loading audio is expensive. check if we can skip this group of utterances.
            all_exist = utterances['audio'].apply(os.path.exists).all()
            if all_exist:
                continue

            logger.debug(f'Writing file {idx} of {num_audios}')
            waveform, sample_rate = librosa.load(original_audio, sr=sample_rate)
            for utterance in utterances.itertuples():
                utterance = utterance._asdict()
                audio = utterance['audio']
                audio_dir = os.path.dirname(audio)
                if os.path.exists(audio):
                    continue

                os.makedirs(audio_dir, exist_ok=True)
                slicer = self._audio_slicer(utterance['offset'], 
                                            utterance['duration'],
                                            sample_rate)
                soundfile.write(audio, 
                                waveform[slicer],
                                sample_rate, 
                                format='flac')
        stop = dt.datetime.now()
        logger.info(f"Writing audio took {stop - start}.")


    def _filter_exists(self, data: pd.DataFrame, path_col: str, log=True):
        """
        Filters out non-existent audio files
        Params:
            data: expects columns {<path_col>}
        """
        unique_paths = pd.Series(data[path_col].unique())
        path_exists = unique_paths.transform(os.path.exists)
        exists_map = dict(zip(unique_paths, path_exists))
        mp3_exists = data[path_col].transform(lambda p: exists_map[p])
        n_missing = mp3_exists.count() - mp3_exists.sum()
        if log:
            logger.info(f'Discarding {n_missing} missing audios.')
        return data.loc[mp3_exists]


    def _filter_empty(self, data: pd.DataFrame, sample_rate: int):
        """
        Filters out non-existent audio files
        Params:
            data: expects columns {duration}
            sample_rate: sample rate in Hz
        """
        if data.empty:
            return data
        not_empty_check = lambda x: x.duration >= self.WINDOW_LEN and \
                                    x.duration * sample_rate > 1
        mp3_notempty = data.apply(lambda x: not_empty_check(x), axis=1)
        num_empty = mp3_notempty.count() - mp3_notempty.sum()
        logger.info(f'Discarding {num_empty} too_short mp3s.')
        return data.loc[mp3_notempty]


    @classmethod
    def _filter_corrupt(cls, data: pd.DataFrame, path_col: str):
        """
        Filters out malformed audio files
        Params:
            data: expects columns {<path_col>}
        """
        unique_paths = pd.Series(data[path_col].unique())
        # Iterating sequentially will be faster than pd.apply 
        # since we dont want to open lots of concurrent file pointers
        corrupt_map = {}
        for unique_path in unique_paths:
            corrupt_map[unique_path] = not cls._is_corrupted(unique_path)
        mp3_notcorrupt = data[path_col].transform(lambda p: corrupt_map[p])
        n_corrupted = mp3_notcorrupt.count() - mp3_notcorrupt.sum()
        logger.info(f'Discarding {n_corrupted} corrupted mp3s')
        return data.loc[mp3_notcorrupt]


    @classmethod
    def _is_corrupted(cls, mp3_path):
        """
        Tests if library can load mp3.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _ = librosa.core.load(mp3_path, sr=None)
            return False
        except:
            return True


    ###################################
    #### Helper methoda for Load  #####
    ###################################


    @classmethod
    def _sample(cls, 
                data: pd.DataFrame,
                qty:Real, 
                units: DataSizeUnit) -> pd.DataFrame:
        """
        Subset the data by duration, count, or ratio.
        Params:
            data: manifest. expects columns {duration}
            qty: amount of data to keep
            units: units for qty. [seconds, count, ratio]
        """
        if units is None:
            return data
        elif units == DataSizeUnit.SECONDS:
            idx = data['duration'].cumsum().searchsorted(qty)
            return data.head(idx)
        elif units == DataSizeUnit.ROW_COUNT:
            return data.head(qty)
        elif units == DataSizeUnit.ROW_FRAC:
            return data.sample(frac=qty, random_state=1234)
        else:
            raise ValueError('Invalid value {} for {}' \
                .format(units, DataSizeUnit))

