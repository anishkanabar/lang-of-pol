import os
import pickle
import logging
from functools import reduce
from logging import Logger
from typing import Any
import numpy as np
import librosa
# from scipy.io import wavfile
from tensorflow import keras

# from google.cloud import storage

logger = logging.getLogger('asr.utils')


def load_data(file_path: str):
    """ Load arbitrary python objects from the pickled file. """
    with open(file_path, mode='rb') as file:
        return pickle.load(file)


def save_data(data: Any, file_path: str):
    """ Save arbitrary python objects in the pickled file. """
    with open(file_path, mode='wb') as file:
        pickle.dump(data, file)


# def download_from_bucket(bucket_name: str, remote_path: str, local_path: str):
#     """ Download the file from the public bucket. """
#     client = storage.Client.create_anonymous_client()
#     bucket = client.bucket(bucket_name)
#     blob = storage.Blob(remote_path, bucket)
#     blob.download_to_filename(local_path, client=client)


# def maybe_download_from_bucket(bucket_name: str, remote_path: str, local_path: str):
#     """ Download file from the bucket if it does not exist. """
#     if os.path.isfile(local_path):
#         return
#     directory = os.path.dirname(local_path)
#     os.makedirs(directory, exist_ok=True)
#     logger.info('Downloading file from the bucket...')
#     download_from_bucket(bucket_name, remote_path, local_path)


def read_audio(file_path: str, sample_rate: int, mono: bool, 
        offset: float = 0, duration: float = 0) -> np.ndarray:
    """ Read already prepared features from the store. """
    # fs, audio = wavfile.read(file_path)
    if duration == 0:
        duration = librosa.core.get_duration(filename=file_path)
    # XXX: Default resampling method is very slow. Trying faster ones.
    # kaiser_fast doesn't load faster but it trains much faster!?
    audio = librosa.core.load(file_path, sr=sample_rate, mono=mono,
            offset=offset, duration=duration)[0]
    return audio

def slice_audio(series: np.ndarray, sample_rate: int,
                offset: float = 0, duration: float = 0) -> np.ndarray:
    offset_idx = librosa.time_to_samples(offset, sr=sample_rate)
    duration_idx = librosa.time_to_samples(offset + duration, sr=sample_rate)
    duration_idx = min(duration_idx, len(series) - 1) # Takes whole audio if dur==0
    return series[offset_idx : duration_idx]

def calculate_units(model: keras.Model) -> int:
    """ Calculate number of the model parameters. """
    units = 0
    for parameters in model.get_weights():
        units += reduce(lambda x, y: x * y, parameters.shape)
    return units


def create_logger(file_path=None, level=20, name='asr') -> Logger:
    """ Create the logger and handlers both console and file. """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] [%(name)-20s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)  # handle all messages from logger
    if file_path:
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
