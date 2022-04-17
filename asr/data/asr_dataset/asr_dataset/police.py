'''
File: police.py
Brief: Loader for police broadcast transcripts and audio files.
Authors: Eric Chandler <echandler@uchicago.edu>
'''

import os
import re
import logging
import pandas as pd
import datetime as dt
from enum import Enum, auto
from numbers import Real
from numpy.random import default_rng
from asr_dataset.base import AsrETL
from asr_dataset.constants import DATASET_DIRS, Cluster, DataSizeUnit

logger = logging.getLogger('asr.etl.bpc')


class AmbiguityStrategy(Enum):
    RANDOM = auto()
    ALL = auto()


class BpcETL(AsrETL):
    """ Data loader for broadcast police communications """
    BAD_WORDS = ["\[UNCERTAIN\]", "<X>", "INAUDIBLE"] # used as regex, thus [] escaped
    SAMPLE_RATE = 22050  # Hz

    def __init__(self, 
                cluster: Cluster=Cluster.RCC, 
                filter_inaudible=True,
                filter_uncertain=True,
                filter_numeric=True,
                ambiguity=AmbiguityStrategy.RANDOM):
        super().__init__('police')
        self.transcripts_dir = DATASET_DIRS[cluster]['police_transcripts']
        self.mp3s_dir = DATASET_DIRS[cluster]['police_mp3s']
        self.filter_inaudible = filter_inaudible
        self.filter_uncertain = filter_uncertain
        self.filter_numeric = filter_numeric
        self.ambiguity_strategy = ambiguity
        

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
        output of this function on disk.
        """
        sample_rate = sample_rate if sample_rate is not None else self.SAMPLE_RATE
        # Filter out bad audio
        unfiltered_data = data
        data = self._filter_exists(data, "original_audio")
        data = self._filter_empty(data, sample_rate)
        #data = self._filter_corrupt(data, "original_audio")
        # Filter out missing transcripts
        missing = (data['text'].isna()) | (data['text'].str.len() == 0)
        logger.info(f'Discarding {missing.sum()} transcripts with no text.')
        data = data.loc[~ missing]
        # Filter out inaudible transcripts
        if self.filter_inaudible:
            has_x = data['text'].str.contains('|'.join(self.BAD_WORDS), regex=True, case=False)
            logger.info(f'Discarding {has_x.sum()} inaudible transcripts.')
            data = data.loc[~ has_x]
        # Filter out uncertain transcripts
        if self.filter_uncertain:
            has_brackets = data['text'].str.contains('\[.+\]', regex=True)
            logger.info(f'Discarding {has_brackets.sum()} uncertain transcripts.')
            data = data.loc[~ has_brackets]
        # Filter out transcripts with numbers
        if self.filter_numeric:
            has_numeric = data['text'].str.contains("[0-9]+", regex=True)
            logger.info(f'Discarding {has_numeric.sum()} transcripts with numerals.')
            data = data.loc[~ has_numeric]
        # Resolve inter-transcriber ambiguity
        # data = self._resolve_ambiguity(data)
        self.describe(data, "-transformed")
        # Write new files to disk
        self._write_utterances(data, sample_rate)
        # Filter out bad audio again after writing
        data = self._filter_exists(data, "audio")
        data = self._filter_empty(data, sample_rate)
        #data = self._filter_corrupt(data, "audio")
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
        

    def _resolve_ambiguity(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handles inter-transcriber label ambiguity.
        Params:
            data: expects columns {original_audio}
        """
        if self.ambiguity_strategy == AmbiguityStrategy.ALL:
            return data

        # Find utterances in same 30m file that overlap in time
        utt_end = data['offset'] + data['duration']
        intervals = [pd.Interval(x,y) for x,y in zip(data['offset'], utt_end)]
        data = data.assign(interval = intervals, end = utt_end)
        candidates = data.merge(data, on='original_audio')
        # remove reverse duplicates: e.g. when (x,y) followed by (y,x)
        candidates = candidates.loc[candidates['offset_x'] <= candidates['offset_y']]
        interval_x = pd.arrays.IntervalArray(candidates['interval_x'])
        interval_y = pd.arrays.IntervalArray(candidates['interval_y'])
        overlaps = [ix.overlaps(iy) for ix, iy in zip(interval_x, interval_y)]
        same_scriber = candidates['transcriber_x'] == candidates['transcriber_y']
        candidates = candidates.loc[overlaps & ~same_scriber]
        logger.debug(f'Found {len(candidates)} that overlap somewhat.')

        # Filter on the amount of time overlap
        OVERLAP_THRESHOLD = .5   # arbitrary
        interval_x = pd.arrays.IntervalArray(candidates['interval_x'])
        interval_y = pd.arrays.IntervalArray(candidates['interval_y'])
        overlap_x = pd.Series(interval_x.right - interval_y.left, index=candidates.index)
        overlap_y = pd.Series(interval_y.right - interval_x.left, index=candidates.index)
        intersect = pd.concat([overlap_x, overlap_y], axis=1).apply(min, axis=1)
        length_x = pd.Series(interval_x.length, index=candidates.index)
        length_y = pd.Series(interval_y.length, index=candidates.index)
        shorter = pd.concat([length_x, length_y], axis=1).apply(min, axis=1)
        overlap = intersect / shorter
        candidates = candidates.loc[overlap > OVERLAP_THRESHOLD]
        logger.debug(f'Found {len(candidates)} that overlap > 50%.')
        
        # Filter on the text similariy
        # XXX: How do you actually measure the within-overlap match without word timings?        
        def _jaccard(str1, str2):
            words1, words2 = str1.split(' '), str2.split(' ')
            bag1, bag2 = set(words1), set(words2)
            return len(bag1 & bag2) / len(bag1 | bag2)
        TEXT_SIM_THRESHOLD = .5  # arbitrary
        sim = candidates.apply(lambda x: _jaccard(x['text_x'],x['text_y']), axis=1)
        candidates = candidates.loc[sim > TEXT_SIM_THRESHOLD]
        logger.debug(f'Found {len(candidates)} that text match > 50%.')

        # Resolve ambiguities by randomly choosing one transcriber's version
        # Challenges:
        # - maintain consistency for interleaved / many-to-many overlaps
        # - give equal chance to each transcriber (n > 2)
        candidate_audios = pd.concat([candidates['audio_x'], candidates['audio_y']])
        ambiguous = data.loc[data['audio'].isin(candidate_audios)]
        non_ambiguous = data.loc[data.index.difference(ambiguous.index)]
        winners = set([])
        losers = set([])
        rng = default_rng()
        for x,y in candidates[['audio_x','audio_y']].sample(len(candidates)).itertuples(index=False):
            if x in winners:
                losers.add(y)
            elif x in losers:
                winners.add(y)
            elif y in winners:
                losers.add(x)
            elif y in losers:
                winners.add(x)
            elif rng.random() > .5:
                winners.add(x)
                losers.add(y)
            else:
                winners.add(y)
                losers.add(x)
        resolved = data.loc[data['audio'].isin(winners)]
        logger.info(f'Discarding {len(losers)} inter-transcriber ambiguous utts.')
        assert len(winners & losers) == 0, f'its {len(winners & losers)}'
        assert len(data) == len(non_ambiguous) + len(resolved) + len(losers)
        return pd.concat([non_ambiguous, resolved])

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
                             'transcriber': self._extract_transcriber(df),
                             'text': df['transcription']})


    def _extract_transcriber(self, df):
        """ Parse transcriber from transcription csv """
        return df['transcriber']


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

