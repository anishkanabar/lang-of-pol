'''
File: police.py
Brief: Loader for police broadcast transcripts and audio files.
Authors: Eric Chandler <echandler@uchicago.edu>
'''

import os
import re
import logging
import datetime as dt
from numbers import Real
import pandas as pd
from asr_dataset.base import AsrETL
from asr_dataset.constants import DATASET_DIRS, Cluster, DataSizeUnit


logger = logging.getLogger('asr.etl.bpc')


class BpcETL(AsrETL):
    """ Data loader for broadcast police communications """
    BAD_WORDS = ["\[UNCERTAIN\]", "<X>", "INAUDIBLE"] # used as regex, thus [] escaped
    SAMPLE_RATE = 22050  # Hz

    def __init__(self, cluster: Cluster=Cluster.RCC):
        super().__init__('police')
        self.transcripts_dir = DATASET_DIRS[cluster]['police_transcripts']
        self.mp3s_dir = DATASET_DIRS[cluster]['police_mp3s']
        

    def extract(self) -> pd.DataFrame:
        """
        Collect info on audio files and transcripts in their original form.

        Returns: DataFrame with columns:
            audio - (str) full path to audio file
            text - (str) transcript text
            duration - (float) length of audio in seconds
        """
        data = self._create_manifest()
        self.describe(data, "-extracted")
        return data


    def transform(self, data: pd.DataFrame, sample_rate: int=None) -> pd.DataFrame:
        """
        Writes new audio files at the utterance level.

        This function is intended to run ONCE over the lifetime of the dataset,
        but is a NO-OP if called again. To re-transform the dataset, delete the
        output of this function.
        """
        sample_rate = sample_rate if sample_rate is not None else self.SAMPLE_RATE
        # Filter out bad audio
        unfiltered_data = data
        data = self._filter_exists(data, "original_audio")
        data = self._filter_empty(data, sample_rate)
        data = self._filter_corrupt(data, "original_audio")
        # Write new files to disk
        self._write_utterances(data, sample_rate)
        # Filter out bad audio again after writing
        data = self._filter_exists(data, "audio")
        data = self._filter_empty(data, sample_rate)
        data = self._filter_corrupt(data, "audio")
        # Filter out missing transcripts
        missing = (data['text'].isna()) | (data['text'].str.len() == 0)
        logger.info(f'Discarding {missing.sum()} transcripts with no text.')
        data = data.loc[~ missing]
        # Filter out inaudible transcripts
        has_x = data['text'].str.contains('|'.join(self.BAD_WORDS), regex=True, case=False)
        logger.info(f'Discarding {has_x.sum()} inaudible transcripts.')
        data = data.loc[~ has_x]
        # Filter out uncertain transcripts
        has_brackets = data['text'].str.contains('\[.+\]', regex=True)
        logger.info(f'Discarding {has_brackets.sum()} uncertain transcripts.')
        data = data.loc[~ has_brackets]
        # Filter out transcripts with numbers
        has_numeric = data['text'].str.contains("[0-9]+", regex=True)
        logger.info(f'Discarding {has_numeric.sum()} transcripts with numerals.')
        data = data.loc[~ has_numeric]
        self.describe(data, "-transformed")
        # Delete bad utterance files
        dropped_data = unfiltered_data.loc[unfiltered_data.index.difference(data.index)]
        for dropped_row in dropped_data.itertuples():
            dropped_row = dropped_row._asdict()
            if os.path.exists(dropped_row['audio']):
                os.remove(dropped_row['audio'])
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
            data = self._create_manifest()
        data = self._filter_exists(data, "audio", log=False)
        data = self._sample(data, qty, units)
        self.describe(data, '-loaded')
        return data 
        

    def _create_manifest(self) -> pd.DataFrame:
        """
        Collect info on audio files and transcripts.
        Returns: DataFrame with columns:
            audio - full path to utterance audio file
            text - transcript text
            duration - audio length in seconds
            original_audio - path to audio file with multiple utterances
        """
        data = self._parse_manifests(self.transcripts_dir)
        data = self._add_utterance_paths(data)
        data = data.drop_duplicates('audio', ignore_index=True)
        return data
    
    
    def _add_utterance_paths(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add column with path to audio utterance.
        Params:
            data: expects columns {original_audio, offset, duration, text}
        """
        msPerSec = 1000
        start_ms = data['offset'].astype(float) * msPerSec
        duration_ms = data['duration'].astype(float) * msPerSec
        end_ms = start_ms + duration_ms
        start_str = start_ms.astype(int).astype('str')  # cast to int for cleaner filenames
        end_str = end_ms.astype(int).astype('str')

        audio_names = start_str.str.cat(end_str, '_') + '.flac'
        audio_paths = data['original_audio'].str.replace('data', 'data/utterances', regex=False)
        audio_paths = audio_paths.str.replace('\.(mp3|wav|flac|ogg)', '/', regex=True)
        audio_paths = audio_paths.str.cat(audio_names)
        return data.assign(audio=audio_paths)
        

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
        return pd.DataFrame({'original_audio': self._extract_mp3_path(df),
                             'offset': self._extract_offset(df),
                             'duration': self._extract_duration(df),
                             'text': df['transcription']})
    

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

