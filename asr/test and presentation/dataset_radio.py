'''
File: dataset_radio.py
Brief: Loader for police radio transcripts and audio files.
Authors: Eric Chandler <echandler@uchicago.edu>
'''

import os
import re
import datetime
import numpy as np
import pandas as pd
import librosa

MP3_DIR = '/project/graziul/data/'

def load_transcripts(transcripts_dir):
    """
    This function is to get audios and transcripts needed for training
    Params:
        @transcripts_dir: path to directory with transcripts csvs
    """
    df = _match_police_audio_transcripts(transcripts_dir)
    df = _clean_transcripts(df)
    df = _clean_audiofiles(df)
    return df


def _clean_transcripts(df, drop_uncertain=True, drop_numeric=True):
    """
    Filters out transcripts with marked uncertain passages
    Params:
        @df: data frame of transcribed utterances
    """
    missing = df['transcripts'].isna()
    print(f'Discarding {missing.sum()} missing transcripts.')
    df = df.loc[~ missing]

    if drop_uncertain:
        has_x = df['transcripts'].str.contains('<x>', case=False)
        print(f'Discarding {has_x.sum()} uncertain transcripts.')
        df = df.loc[~ has_x]

    if drop_numeric:
        has_numeric = df['transcripts'].str.contains("[0-9]")
        print(f'Discarding {has_numeric.sum()} transcripts with numbers.')
        df = df.loc[~ has_numeric]
        
    return df


def _clean_audiofiles(df):
    """
    Filters out non-existent and corrupted mp3's
    Params:
        @df: data frame of mp3 filepaths
    """
    unique_paths = pd.Series(df['path'].unique())
    path_exists = unique_paths.transform(os.path.exists)
    exists_map = {x:y for x,y in zip(unique_paths, path_exists)}

    mp3_exists = df['path'].transform(lambda p: exists_map[p])
    n_missing = mp3_exists.count() - mp3_exists.sum()
    
    validator = lambda p: exists_map[p] and not _is_corrupted(p)
    path_isvalid = unique_paths.transform(validator)
    valid_map = {x:y for x,y in zip(unique_paths, path_isvalid)}

    mp3_isvalid = df['path'].transform(lambda p: valid_map[p])
    n_corrupted = mp3_isvalid.count() - mp3_isvalid.sum() - n_missing
    print(f'Discarding {n_missing} missing mp3s')
    print(f'Discarding {n_corrupted} corrupted mp3s')
    df = df.loc[mp3_exists & mp3_isvalid]
    return df

def _is_corrupted(mp3_path):
    """
    Tests if library can load mp3.
    """
    try:
        _ = librosa.core.load(mp3_path)
        return False
    except:
        return True


def _match_police_audio_transcripts(ts_dir):
    """
    Matches ~second-long transcripts to ~30minute source audio file.
    Params:
        @ts_dir: Location of folder with transcripts csvs
    """
    ts_dir_files = os.listdir(ts_dir)
    pattern = "transcripts\d{4}_\d{2}_\d{2}.csv"
    ts_names = [x for x in ts_dir_files if re.match(pattern, x)]
    ts_paths = [os.path.join(ts_dir, x) for x in ts_names]
    audio_dfs = [_match_utterance_to_audio(x) for x in ts_paths]
    return pd.concat(audio_dfs, ignore_index=True)


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
        
         
