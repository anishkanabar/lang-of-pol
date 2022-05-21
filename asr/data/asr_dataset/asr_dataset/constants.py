"""
File: constants.py
Brief: environmet variables and common enums
Authors: Eric Chandler <echandler@uchicago.edu>
"""
from enum import Enum, auto


class Cluster(Enum):
    RCC = auto()
    AI = auto()
    TTIC = auto()


DATASET_DIRS = {
    Cluster.RCC: {
        'police_transcripts':'/project/graziul/transcripts',
        'police_mp3s':'/project/graziul/data',
        'librispeech_data':'/project/graziul/ra/shiyanglai/experiment1/audio-data/LibriSpeech/train-clean-100',
        'atczero_data' : '/project/graziul/ra/wdolan/atc0_comp',
    },
    Cluster.AI: {
        'police_transcripts':'/net/projects/uri/transcripts',
        'police_mp3s':'/net/projects/uri/data',
        'librispeech_data':'/net/projects/uri/ra/shiyanglai/experiment1/audio data/LibriSpeech/train-clean-100',
        'atczero_data': '/net/projects/uri/ra/wdolan/atc0_comp',
    },
    Cluster.TTIC: {
        'librispeech_data': '/share/data/speech/Datasets/LibriSpeech/LibriSpeech/train-clean-100',
        'atczero_data': '/share/data/speech/Data/echandler/corpora/atc0_comp',
    }
}
