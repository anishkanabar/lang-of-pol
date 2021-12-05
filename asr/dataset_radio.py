'''
File: dataset_radio.py
Brief: Loader for police radio transcripts and audio files.
Authors: Eric Chandler <echandler@uchicago.edu>
'''

import os
import re
import datetime
import pandas as pd
import logging
import pathlib
from dataset import AudioClipDataset

MP3_DIR = '/project/graziul/data/'
#MP3_DIR = '/Users/eric/Documents/Work/PoliceBroadcasts/FakeData/'
BAD_WORDS = ["\[UNCERTAIN\]", "<X>", "INAUDIBLE"] # used as regex, thus [] escaped
SAMPLE_RATE = 16000  # Hz
WINDOW_LEN = .02 # Sec

class RadioDataset(AudioClipDataset):
    
    @classmethod
    def load_transcripts(cls, transcripts_dir:pathlib.Path, 
                              sample_rate=SAMPLE_RATE, 
                              window_len=WINDOW_LEN,
                              drop_bad_audio=True, 
                              drop_inaudible=True, 
                              drop_uncertain=True, 
                              drop_numeric=True):
        """
        This function is to get audios and transcripts needed for training
        Params:
            @transcripts_dir: path to directory with transcripts csvs
            @sample_rate: Resample all audio to this rate (Hz)
            @window_len: Effective minimum audio length (sec)
            @drop_bad_audio: Filter out missing/corrupted/too-short audio files
            @drop_inaudible: Filter out utterances that the transcriber couldnt understand
            @drop_uncertain: Filter out utterances that the transcriber was guessing
            @drop_numeric: Filter out utterances transcribed with 0-9 instead of pronunciation
        """
        df = _match_police_audio_transcripts(transcripts_dir)
        df = _add_clip_paths(df)
        logging.info(f"Original dataset has {df.shape[0]} rows.")
        df = _filter_transcripts(df, drop_inaudible, drop_uncertain, drop_numeric)
        if drop_bad_audio:
            df = cls.filter_audiofiles(df, sample_rate, window_len)
        cls.describe(df, "Loaded")
        return df
    
    
def _filter_transcripts(df, drop_inaudible=True, drop_uncertain=True, drop_numeric=True):
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


def _match_police_audio_transcripts(ts_dir):
    """
    Matches ~second-long transcripts to ~30minute source audio file.
    Params:
        @ts_dir: Location of folder with transcripts csvs
    """
    ts_dir_files = pathlib.iterdir(ts_dir)
    pattern = "transcripts\d{4}_\d{2}_\d{2}.csv"
    ts_names = [x for x in ts_dir_files if re.match(pattern, x)]
    ts_paths = [os.path.join(ts_dir, x) for x in ts_names]
    audio_dfs = [_match_utterance_to_audio(x) for x in ts_paths]
    return pd.concat(audio_dfs, ignore_index=True)


def _add_clip_paths(data):
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
    

def _match_utterance_to_audio(ts_path):
    """
    Extracts mp3 path, utterance timestamp, and duration from transcript metadata
    Params:
        @ts_path: Location of transcript csv
    """
    df = pd.read_csv(ts_path)
    return pd.DataFrame({'path': _extract_mp3_path(df),
                         'offset': _extract_offset(df),
                         'duration': _extract_duration(df),
                         'transcripts': df['transcription']})

def _extract_mp3_path(df):
    """
    Parse mp3 file path for utterance in transcription csv
    """
    year = df['file'].str.extract("(\d{4})", expand=False)
    month = df['file'].str.extract("\d{4}(\d{2})", expand=False)
    day = df['file'].str.extract("\d{6}(\d{2})", expand=False)
    date = year.str.cat([month, day], sep="_")
    name = df['file'].str.extract("(\d+-\d+-\d+)", expand=False) + ".mp3"
    return MP3_DIR + df['zone'] + '/' + date + '/' + name
    

def _extract_offset(df):
    """
    Parses the utterance start time from transcription csv
    """
    origin = datetime.datetime(1900, 1, 1)
    offset = pd.to_datetime(df['start_dt']) - origin
    return offset.dt.total_seconds()

def _extract_duration(df):
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
        
         
