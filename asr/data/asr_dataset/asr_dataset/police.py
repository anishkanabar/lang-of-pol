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
from asr_dataset.constants import DATASET_DIRS, Cluster

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
                ambiguity: str='RANDOM'):
        super().__init__('police')
        self.transcripts_dir = DATASET_DIRS[cluster]['police_transcripts']
        self.mp3s_dir = DATASET_DIRS[cluster]['police_mp3s']
        self.filter_inaudible = filter_inaudible
        self.filter_uncertain = filter_uncertain
        self.filter_numeric = filter_numeric
        self.ambiguity_strategy = AmbiguityStrategy[ambiguity.upper()]
        

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
        # data = self._filter_corrupt(data, "original_audio")
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
        data = self._resolve_ambiguity(data)
        self.describe(data, "-transformed")
        # Write new files to disk
        self._write_utterances(data, sample_rate)
        # Filter out bad audio again after writing
        data = self._filter_exists(data, "audio")
        data = self._filter_empty(data, sample_rate)
        data = self._filter_corrupt(data, "audio")
        return data

    
    def load(self,
            data: pd.DataFrame=None,
            splits: dict=None,
            seed: int=1234) -> pd.DataFrame:
        """
        Collect info on the transformed audio files and transcripts.
        Does NOT load waveforms into memory.

        Returns: DataFrame with same columns as extract()
        """
        if data is None:
            data = self._create_manifest()
        data = self._filter_exists(data, "audio", log=False)
        data = self._sample_split(data, splits, seed)
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
        
    def _resolve_overlaps(self, data: pd.DataFrame) -> set([str]):
        """
        Finds maximal number of non-overlapping intervals with random tie-breaks.
        Param:
            data: expects columns {audio_x, audio_y, offset_x, offset_y, end_x, end_y}
        """
        data = data.sort_values(['end_x','end_y'])
        keep = set([])
        counter = -1
        rnd = default_rng()
        for tup in data.itertuples(index=False):
            tup = tup._asdict()
            ax, ay = tup['audio_x'], tup['audio_y']    # keys
            ex, ey = tup['end_x'], tup['end_y']        # ordering maintains consistency
            ox, oy = tup['offset_x'], tup['offset_y']  # determines if fits in 'keep' set
            if ox > counter and oy > counter:
                if rnd.random() > 0.5:
                    keep.add(ax)
                    counter = ex
                else:
                    keep.add(ay)
                    counter = ey
            elif ox > counter:
                keep.add(ax)
                counter = ex
            elif oy > counter:
                keep.add(ay)
                counter = ey
        return keep


    def _resolve_ambiguity(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handles inter-transcriber label ambiguity.
        Params:
            data: expects columns {original_audio}
        """
        # Do this in all cases so columns align
        data = data.assign(end = data['offset'] + data['duration'])

        if self.ambiguity_strategy == AmbiguityStrategy.ALL:
            return data

        # Find utterances in same 30m file that overlap in time
        candidates = data.merge(data, on='original_audio')
        # remove "reverse duplicates": e.g. when row 1 = (x,y) and row 2 = (y,x)
        # by keeping the version where x is on the left of the join
        # conveniently this makes computing overlap easier
        overlaps = (candidates['offset_x'] <= candidates['offset_y']) \
                    & (candidates['end_x'] >= candidates['offset_y'])
        same_scriber = candidates['transcriber_x'] == candidates['transcriber_y']
        candidates = candidates.loc[overlaps & ~same_scriber]
        n_candidates = pd.concat([candidates['audio_x'],candidates['audio_y']]).nunique()
        logger.debug(f'Found {n_candidates} that overlap somewhat.')

        # Pandas column-wise aggregation has unusual behavior for 0-length data-frames. 
        if n_candidates == 0:
            return data

        # Filter on the amount of time overlap
        OVERLAP_THRESHOLD = .5   # arbitrary
        intersect = candidates['end_x'] - candidates['offset_y']
        overlap = pd.concat([candidates['duration_y'], intersect], axis=1).apply(min, axis=1)
        shorter = candidates[['duration_x','duration_y']].apply(min, axis=1)
        candidates = candidates.loc[(overlap / shorter) > OVERLAP_THRESHOLD]
        n_candidates = pd.concat([candidates['audio_x'],candidates['audio_y']]).nunique()
        logger.debug(f'Found {n_candidates} that overlap > 50%.')
        
        # Filter on the text similariy
        # XXX: How do you actually measure the within-overlap match without word timings?        
        def _jaccard(str1, str2):
            words1, words2 = str1.split(' '), str2.split(' ')
            bag1, bag2 = set(words1), set(words2)
            return 1.0 * len(bag1 & bag2) / len(bag1 | bag2)

        TEXT_SIM_THRESHOLD = .5  # arbitrary
        sim = candidates.apply(lambda x: _jaccard(x['text_x'],x['text_y']), axis=1)
        candidates = candidates.loc[sim > TEXT_SIM_THRESHOLD]
        n_candidates = pd.concat([candidates['audio_x'],candidates['audio_y']]).nunique()
        logger.debug(f'Found {n_candidates} that text match > 50%.')

        # Resolve ambiguities by randomly choosing one transcriber's version
        # Challenges:
        # - maintain consistency for interleaved / many-to-many overlaps
        # - give equal chance to each transcriber (n > 2): satisfied if no patterns in timing
        candidate_audios = pd.concat([candidates['audio_x'], candidates['audio_y']])
        ambiguous = data.loc[data['audio'].isin(candidate_audios)]
        non_ambiguous = data.loc[data.index.difference(ambiguous.index)]
        resolved = data.loc[data['audio'].isin(self._resolve_overlaps(candidates))]
        num_drop = len(data) - len(non_ambiguous) - len(resolved)
        logger.info(f'Discarding {num_drop} inter-transcriber ambiguous utts.')
        return pd.concat([non_ambiguous, resolved])

    def _parse_manifests(self, ts_dir: str) -> pd.DataFrame:
        """
        Matches ~second-long transcripts to ~30minute source audio file.
        Params:
            ts_dir: Location of folder with transcripts csvs
        """
        ts_dir_files = os.listdir(ts_dir)
        pattern = "transcripts2022_02_06.csv"
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

