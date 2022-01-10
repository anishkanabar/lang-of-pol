'''
File: radio.py
Brief: Loader for police radio transcripts and audio files.
Authors: Eric Chandler <echandler@uchicago.edu>
'''

import os
import re
import datetime
import pandas as pd
import logging
from asr_dataset.dataset import AudioClipDataset
from asr_dataset.datasets.constants import DATASET_DIRS

BAD_WORDS = ["\[UNCERTAIN\]", "<X>", "INAUDIBLE"] # used as regex, thus [] escaped
SAMPLE_RATE = 16000  # Hz
WINDOW_LEN = .02 # Sec

class RadioDataset(AudioClipDataset):

    def __init__(self, cluster:str='rcc', nrow: int=None, frac: float=None, window_len=WINDOW_LEN):
        self.transcripts_dir = DATASET_DIRS[cluster]['radio_transcripts']
        self.mp3s_dir = DATASET_DIRS[cluster]['radio_mp3s']
        super().__init__('radio', nrow, frac, window_len)
    
    def _load_transcripts(self, 
                        sample_rate=SAMPLE_RATE, 
                        window_len=WINDOW_LEN,
                        drop_bad_audio=True,
                        drop_inaudible=True, 
                        drop_uncertain=True, 
                        drop_numeric=True):
        """
        This function is to get audios and transcripts needed for training
        Params:
            @sample_rate: Resample all audio to this rate (Hz)
            @window_len: Minimum meaningful audio length (sec)
            @drop_bad_audio: Filter out missing or corrupted or too short audio
            @drop_inaudible: Filter out utterances that the transcriber couldnt understand
            @drop_uncertain: Filter out utterances that the transcriber was guessing
            @drop_numeric: Filter out utterances transcribed with 0-9 instead of pronunciation
        """
        df = self._match_police_audio_transcripts(self.transcripts_dir)
        df = self._add_clip_paths(df)
        logging.debug(f"Original dataset has {df.shape[0]} rows.")
        df = self._filter_transcripts(df, drop_inaudible, drop_uncertain, drop_numeric)
        logging.debug(f"Dataset after transcript filtering has {df.shape[0]} rows.")
        if drop_bad_audio:
            df = self._filter_audiofiles(df, sample_rate, window_len)
            logging.debug(f"Dataset after audio filtering has {df.shape[0]} rows.")
        return df
    
    
    def _filter_transcripts(self, df, drop_inaudible=True, drop_uncertain=True, drop_numeric=True):
        """
        Filters out transcripts with marked uncertain passages
        Params:
            @df: data frame of transcribed utterances
            @drop_inaudible: filter out utterances that the transcriber couldnt understand
            @drop_uncertain: filter out utterances that the transcriber was guessing
            @drop_numeric: filter out utterances transcribed with 0-9 instead of pronunciation
        """
        missing = df['transcripts'].isna()
        logging.info(f'Discarding {missing.sum()} missing transcripts.')
        df = df.loc[~ missing]
    
        if drop_inaudible:
            has_x = df['transcripts'].str.contains('|'.join(BAD_WORDS), regex=True, case=False)
            logging.info(f'Discarding {has_x.sum()} inaudible transcripts.')
            df = df.loc[~ has_x]
    
        if drop_uncertain:
            has_brackets = df['transcripts'].str.contains('\[.+\]', regex=True)
            logging.info(f'Discarding {has_brackets.sum()} uncertain transcripts.')
            df = df.loc[~ has_brackets]
    
        if drop_numeric:
            has_numeric = df['transcripts'].str.contains("[0-9]+", regex=True)
            logging.info(f'Discarding {has_numeric.sum()} transcripts with numerals.')
            df = df.loc[~ has_numeric]
            
        return df
    
    
    def _match_police_audio_transcripts(self, ts_dir):
        """
        Matches ~second-long transcripts to ~30minute source audio file.
        Params:
            @ts_dir: Location of folder with transcripts csvs
        """
        ts_dir_files = os.listdir(ts_dir)
        pattern = "transcripts\d{4}_\d{2}_\d{2}.csv"
        ts_names = [x for x in ts_dir_files if re.match(pattern, x)]
        ts_paths = [os.path.join(ts_dir, x) for x in ts_names]
        audio_dfs = [self._match_utterance_to_audio(x) for x in ts_paths]
        return pd.concat(audio_dfs, ignore_index=True)
    
    
    def _add_clip_paths(self, data):
        """
        Add column with path to audio clip.
        Params:
            @data: expects columns {path, offset, duration, transcripts}
        """
        msPerSec = 1000
        off_fmt = (data['offset'] * msPerSec).astype('int32').astype('str')
        end_fmt = ((data['offset'] + data['duration']) * msPerSec).astype('int32').astype('str')
        ext_fmt = pd.Series(['.flac']*len(data))
        clip_names = off_fmt.str.cat(end_fmt, '_').str.cat(ext_fmt)
        clip_paths = data['path'].str.replace('data','data/utterances') \
                    .str.replace('.mp3', '').str.cat(clip_names, '/')
        return data.assign(clip_path=clip_paths)
        
    
    def _match_utterance_to_audio(self, ts_path):
        """
        Extracts mp3 path, utterance timestamp, and duration from transcript metadata
        Params:
            @ts_path: Location of transcript csv
        """
        df = pd.read_csv(ts_path)
        return pd.DataFrame({'path': self._extract_mp3_path(df),
                             'offset': self._extract_offset(df),
                             'duration': self._extract_duration(df),
                             'transcripts': df['transcription']})
    
    def _extract_mp3_path(self, df):
        """
        Parse mp3 file path for utterance in transcription csv
        """
        year = df['file'].str.extract("(\d{4})", expand=False)
        month = df['file'].str.extract("\d{4}(\d{2})", expand=False)
        day = df['file'].str.extract("\d{6}(\d{2})", expand=False)
        date = year.str.cat([month, day], sep="_")
        name = df['file'].str.extract("(\d+-\d+-\d+)", expand=False) + ".mp3"
        # Don't use os.path.join here since some components are pandas series
        return self.mp3s_dir + os.sep + df['zone'] + os.sep + date + os.sep + name
        
    
    def _extract_offset(self, df):
        """
        Parses the utterance start time from transcription csv
        """
        origin = datetime.datetime(1900, 1, 1)
        offset = pd.to_datetime(df['start_dt']) - origin
        return offset.dt.total_seconds()
    
    def _extract_duration(self, df):
        """
        Parses utterance duration from the transcription csv. Handles inconsisent column formatting
        """
        if 'length_s' in df.columns:
            return df['length_s']
        elif pd.api.types.is_numeric_dtype(df['length'].dtype):
            return df['length']
        else:
            pattern = "%H:%M:%S.%f"
            prefix = "0 days "
            origin = datetime.datetime(1900, 1, 1)
            length_time = df['length'].str.replace(prefix, "")
            length_iso = length_time.str.slice(0, len(pattern))
            length_dt = pd.to_datetime(length_iso, format=pattern)
            length_sec = (length_dt - origin).dt.total_seconds()
            return length_sec
            
             
