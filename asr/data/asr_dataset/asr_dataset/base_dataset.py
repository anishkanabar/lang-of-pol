'''
File: base_dataset.py
Brief: Abstract base class for ASR dataset loaders
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

SAMPLE_RATE = 16000  # Hz
WINDOW_LEN = .02 # sec

logger = logging.getLogger('asr.dataset')

class ASRDataset(abc.ABC):

    def __init__(self, name: str, 
                       nrow: int=None, 
                       frac: float=None, 
                       window_len=WINDOW_LEN):
        data = self._load_transcripts(window_len=WINDOW_LEN)
        if nrow is not None:
            self.data = data.head(nrow)
        elif frac is not None:
            self.data = data.sample(frac=frac, random_state=1234)
        else:
            self.data = data
        self.describe(self.data, name)
        
    @classmethod
    def describe(cls, data: pd.DataFrame, name: str):
        """
        Prints helpful statistics about dataset.
        Params
            @data: Fully loaded transcripts dataframe
        """
        logger.info(f"{name} dataset stats:")
        logger.info(f"\tRow count = {data.shape[0]}")
    
    @abc.abstractmethod
    def _load_transcripts(self, window_len:float):
        """
            This function is to get audios and transcripts needed for training
            Returns: Dataframe of at least path, transcripts
        """
        pass

    @classmethod
    def add_duration(cls, data):
        if 'duration' in data.columns:
            return data
        else:
            duration_func = lambda x: librosa.get_duration(filename=x)
            return data.assign(duration=data['path'].apply(duration_func))

    @classmethod
    def add_sample_count(cls, data):
        if 'nsamples' in data.columns:
            return data
        else:
            data_ = data.assign(sr=data['path'].apply(librosa.get_samplerate))
            count_func = lambda row: librosa.time_to_samples(row['duration'], row['sr'])
            sample_count = data_.apply(lambda row: count_func(row), axis=1)
            return data.assign(nsamples=sample_count)
    
            
class AudioClipDataset(ASRDataset):

    def __init__(self, 
                 name: str, 
                 nrow: int=None, 
                 frac: float=None,
                 window_len=WINDOW_LEN):
        super().__init__(name, nrow=nrow, frac=frac, window_len=window_len)
        self.write_clips(self.data)
        # must re-filter in case new mp3 clips are bad
        self.data = self._filter_corrupt_audio(self.data)
        self.describe(self.data, name)
        

    @classmethod
    def audio_slicer(cls, offset: float, duration: float, sample_rate: int) -> slice:
        offset_idx = librosa.time_to_samples(offset, sr=sample_rate)
        duration_idx = librosa.time_to_samples(offset + duration, sr=sample_rate)
        return slice(offset_idx, duration_idx)
    
    # TODO: Either use new_sample_rate here or get rid of it.
    @classmethod
    def write_clips(cls, data: pd.DataFrame):
        """ 
        Extract small audio clips from original file and write them to disk.
        Params:
            @data: expects columns {path, clip_path, offset, duration, transcripts}
        """
        logger.info('Writing audio clips.')
        start = dt.datetime.now()
        audio_paths = set(data['path'])
        for audio_path in audio_paths:
            clips = data.loc[data['path'] == audio_path]
            # loading audio is expensive. check if we can skip this group of clips.
            all_exist = clips['clip_path'].apply(os.path.exists).all()
            if all_exist:
                continue
            audio_array, sample_rate = librosa.load(audio_path, sr=None)
            for clip in clips.itertuples():
                if os.path.exists(clip.clip_path):
                    #logger.debug(f"File {clip.clip_path} exists. Not overwriting.") 
                    continue
                if not os.path.exists(os.path.dirname(clip.clip_path)):
                    os.makedirs(os.path.dirname(clip.clip_path), exist_ok=True)
                slicer = cls.audio_slicer(clip.offset, clip.duration, sample_rate)
                clip_array = audio_array[slicer]
                # XXX: Throwing system error on AI cluster.
                #      Stacktrace shows RuntimeError error opening path to flac: System error:
                #      in soundfile.py: lines 314 -> 629 -> 1183 -> 1357
                #      But the issue doesn't seem to exist in Midway3
                soundfile.write(clip.clip_path, clip_array, sample_rate, format='flac')
        # Mutate original df!
        data['original_path'] = data['path']
        data['path'] = data['clip_path'] 
        data.drop('clip_path', axis=1, inplace=True)
        stop = dt.datetime.now()
        logger.info(f"Writing audio took {stop - start}.")
    
    @classmethod
    def describe(cls, data, name):
        """
        Prints helpful statistics about dataset.
        Params
            @data: Fully loaded transcripts dataframe
        """
        super().describe(data, name)
        logger.info(f"\tMin duration = {data['duration'].min():.2f}")
        logger.info(f"\tMax duration = {data['duration'].max():.2f}")
        logger.info(f"\tMean duration = {data['duration'].mean():.2f}")
        logger.info(f"\tStdev duration = {data['duration'].std():.2f}") 
        logger.info(f"\tTotal duration = {pd.Timedelta(data['duration'].sum(),'sec')}") 

    @classmethod
    def _filter_audiofiles(cls, df, new_sample_rate=SAMPLE_RATE, window_len=WINDOW_LEN):
        """
        Filters out non-existent and corrupted mp3's
        Params:
            @df: data frame of mp3 filepaths
        """
        unique_paths = pd.Series(df['path'].unique())
        path_exists = unique_paths.transform(os.path.exists)
        exists_map = dict(zip(unique_paths, path_exists))
        mp3_exists = df['path'].transform(lambda p: exists_map[p])
        n_missing = mp3_exists.count() - mp3_exists.sum()
        df = df.loc[mp3_exists]
        logger.info(f'Discarding {n_missing} missing mp3s.')
        
        # Commented out because not needed on main dataset. Only needed on clips.
        #df = cls._filter_corrupt_audio(df)

        not_empty_check = lambda x: x.duration >= window_len or x.duration * new_sample_rate >= 1
        mp3_notempty = df.apply(lambda x: not_empty_check(x), axis=1)
        num_empty = mp3_notempty.count() - mp3_notempty.sum()
        logger.info(f'Discarding {num_empty} too_short mp3s.')
        df = df.loc[mp3_notempty]
        return df

    @classmethod
    def _filter_corrupt_audio(cls, df):
        unique_paths = pd.Series(df['path'].unique())
        path_notcorrupt = unique_paths.transform(lambda p: not cls._is_corrupted(p))
        corrupt_map = dict(zip(unique_paths, path_notcorrupt))
        mp3_notcorrupt = df['path'].transform(lambda p: corrupt_map[p])
        n_corrupted = mp3_notcorrupt.count() - mp3_notcorrupt.sum()
        logger.info(f'Discarding {n_corrupted} corrupted mp3s')
        df = df.loc[mp3_notcorrupt]
        return df
    
    @classmethod
    def _is_corrupted(cls, mp3_path):
        """
        Tests if library can load mp3.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _ = librosa.core.load(mp3_path)
            return False
        except:
            return True
