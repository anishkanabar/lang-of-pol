'''
File: utterance_dataset.py
Brief: Base class for ASR datasets that must be converted into utterances.
Authors: Eric Chandler <echandler@uchicago.edu>
'''

import os
import abc
import warnings
import datetime as dt
import pandas as pd
import logging
import librosa
import soundfile
from asr_dataset.base_dataset import ASRDataset

logger = logging.getLogger('asr.dataset.utterance')

class UtteranceDataset(ASRDataset):
    SAMPLE_RATE = 16000  # Hz
    WINDOW_LEN = .04 # sec

    def __init__(self, 
                 name: str, 
                 nrow: int=None, 
                 frac: float=None,
                 nsecs: float=None,
                 resample: int=None):
        self.overwrite = resample is not None
        self.new_sample_rate = resample if self.overwrite else self.SAMPLE_RATE
        super().__init__(name, nrow=nrow, frac=frac, nsecs=nsecs)
        if self.overwrite:
            self._write_utterance_audios()
        # must re-filter in case new mp3 clips are bad
        self.data = self._filter_exists(self.data, 'path')
        self.data = self._filter_corrupt(self.data, 'path')
        self.describe()
    
    @abc.abstractmethod
    def create_manifest(self) -> pd.DataFrame:
        """
        Collect info on audio files and transcripts.
        Returns: DataFrame with columns:
            path - full path to utterance audio file
            transcripts - transcript text
            duration - length of audio in (float) seconds
            context_path - full path to audio file w/ multiple utterances
        """
        pass

    def _write_utterance_audios(self):
        """ 
        Extract small audio clips from original file and write them to disk.
        Params:
            data: expects columns {path, transcripts, duration, context_path}
        """
        logger.info('Writing utterance audio clips.')
        start = dt.datetime.now()
        context_paths = set(self.data['context_path'])
        for idx, context_path in enumerate(context_paths):
            logger.debug(f'Writing file {idx} of {len(context_paths)}')
            utterances = self.data.loc[self.data['context_path'] == context_path]
            # loading audio is expensive. check if we can skip this group of utterances.
            all_exist = utterances['path'].apply(os.path.exists).all()
            if all_exist and not self.overwrite:
                continue
            audio_array, sample_rate = librosa.load(context_path, sr = self.new_sample_rate)
            for utterance in utterances.itertuples():
                if os.path.exists(utterance.path) and not self.overwrite:
                    #logger.debug(f"File {utterance.path} exists. Not overwriting.") 
                    continue
                if not os.path.exists(os.path.dirname(utterance.path)):
                    os.makedirs(os.path.dirname(utterance.path), exist_ok=True)
                slicer = self._audio_slicer(utterance.offset, utterance.duration, sample_rate)
                utterance_array = audio_array[slicer]
                # XXX: Throwing system error on AI cluster.
                #      Stacktrace shows RuntimeError error opening path to flac: System error:
                #      in soundfile.py: lines 314 -> 629 -> 1183 -> 1357
                #      But the issue doesn't seem to exist in Midway3
                soundfile.write(utterance.path, utterance_array, sample_rate, format='flac')
        stop = dt.datetime.now()
        logger.info(f"Writing audio took {stop - start}.")
    

    @classmethod
    def _audio_slicer(cls, offset: float, duration: float, sample_rate: int) -> slice:
        offset_idx = librosa.time_to_samples(offset, sr=sample_rate)
        duration_idx = librosa.time_to_samples(offset + duration, sr=sample_rate)
        return slice(offset_idx, duration_idx)

    def filter_manifest(self, data: pd.DataFrame):
        """
        Filters out non-existent and too-short audios
        Params:
            data: expects columns {context_path, duration}
        """
        data = self._filter_exists(data, 'context_path')
        data = self._filter_empty(data)
        return data


    def _filter_exists(self, data: pd.DataFrame, path_col: str):
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
        logger.info(f'Discarding {n_missing} missing mp3s.')
        return data.loc[mp3_exists]


    def _filter_empty(self, data: pd.DataFrame):
        """
        Filters out non-existent audio files
        Params:
            data: expects columns {duration}
        """
        not_empty_check = lambda x: x.duration >= self.WINDOW_LEN and \
                                    x.duration * self.new_sample_rate > 1
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
        path_notcorrupt = unique_paths.transform(lambda p: not cls._is_corrupted(p))
        corrupt_map = dict(zip(unique_paths, path_notcorrupt))
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
