'''
File: police.py
Brief: Loader for police broadcast transcripts and audio files.
Authors: Eric Chandler <echandler@uchicago.edu>
'''

import os
import re
import datetime as dt
import pandas as pd
import logging
from asr_dataset.utterance_dataset import UtteranceDataset
from asr_dataset.constants import DATASET_DIRS

logger = logging.getLogger('asr.dataset.police')

BAD_WORDS = ["\[UNCERTAIN\]", "<X>", "INAUDIBLE"] # used as regex, thus [] escaped

class PoliceDataset(UtteranceDataset):

    SAMPLE_RATE = 22050  # Hz

    def __init__(self, 
                 cluster:str='rcc', 
                 nrow: int=None, 
                 frac: float=None, 
                 nsecs: float=None,
                 resample: int=None):
        self.transcripts_dir = DATASET_DIRS[cluster]['police_transcripts']
        self.mp3s_dir = DATASET_DIRS[cluster]['police_mp3s']
        super().__init__('police', nrow, frac, nsecs, resample)
    

    def filter_manifest(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out non-existent, too-short, and inaudible audios
        Params:
            data: expects columns {context_path, duration, transcripts}
        """
        data = super().filter_manifest(data)

        missing = data['transcripts'].isna()
        logger.info(f'Discarding {missing.sum()} transcripts with no text.')
        data = data.loc[~ missing]
    
        has_x = data['transcripts'].str.contains('|'.join(BAD_WORDS), regex=True, case=False)
        logger.info(f'Discarding {has_x.sum()} inaudible transcripts.')
        data = data.loc[~ has_x]
    
        has_brackets = data['transcripts'].str.contains('\[.+\]', regex=True)
        logger.info(f'Discarding {has_brackets.sum()} uncertain transcripts.')
        data = data.loc[~ has_brackets]
    
        has_numeric = data['transcripts'].str.contains("[0-9]+", regex=True)
        logger.info(f'Discarding {has_numeric.sum()} transcripts with numerals.')
        data = data.loc[~ has_numeric]
            
        return data

    def create_manifest(self) -> pd.DataFrame:
        """
        Collect info on audio files and transcripts.
        Returns: DataFrame with columns:
            path - full path to utterance audio file
            transcripts - transcript text
            duration - audio length in seconds
            context_path - path to audio file with multiple utterances
        """
        data = self._parse_manifests(self.transcripts_dir)
        data = self._add_utterance_paths(data)
        data = data.drop_duplicates('path', ignore_index=True)
        return data
    
    
    def _add_utterance_paths(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add column with path to audio utterance.
        Params:
            data: expects columns {context_path, offset, duration, transcripts}
        """
        msPerSec = 1000
        off_fmt = (data['offset'] * msPerSec).astype('int32').astype('str')
        end_fmt = ((data['offset'] + data['duration']) * msPerSec).astype('int32').astype('str')
        ext_fmt = pd.Series(['.flac']*len(data))
        utterance_names = off_fmt.str.cat(end_fmt, '_').str.cat(ext_fmt)
        utterance_paths = data['context_path'].str.replace('data','data/utterances', regex=False) \
                    .str.replace('.mp3', '').str.cat(utterance_names, '/')
        return data.assign(path=utterance_paths)
        

    def _parse_manifests(self, ts_dir: str) -> pd.DataFrame:
        """
        Matches ~second-long transcripts to ~30minute source audio file.
        Params:
            ts_dir: Location of folder with transcripts csvs
        """
        ts_dir_files = os.listdir(ts_dir)
        pattern = "transcripts\d{4}_\d{2}_\d{2}.csv"
        ts_names = [x for x in ts_dir_files if re.match(pattern, x)]
        ts_paths = [os.path.join(ts_dir, x) for x in ts_names]
        audio_dfs = [self._parse_csv(x) for x in ts_paths]
        return pd.concat(audio_dfs, ignore_index=True)
    
    
    def _parse_csv(self, ts_path: str) -> pd.DataFrame:
        """
        Extracts mp3 path, utterance timestamp, and duration from transcript metadata
        Params:
            ts_path: Location of transcript csv
        """
        df = pd.read_csv(ts_path)
        return pd.DataFrame({'context_path': self._extract_mp3_path(df),
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
        origin = dt.datetime(1900, 1, 1)
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
            delta = pd.to_datetime(df['end_dt']) - pd.to_datetime(df['start_dt'])
            return delta.dt.total_seconds()

