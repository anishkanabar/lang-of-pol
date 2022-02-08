"""
speaker diarization framework
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

import argparse
import pytorch_lightning as pl
from pyannote.audio import Pipeline
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.models.embedding import XVectorMFCC
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
from utils import generate_ramdom_code
from data_format import *
from evaluation import *
from tunning import *
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    # parse the command line
    parser = argparse.ArgumentParser(description='Speaker diarization pipeline of the framework.')
    parser.add_argument('corpus_name', default='ami', help='the name of the corpus (default: AMI)')
    parser.add_argument('pretrained', default=0, help='load pretrained models or not.')
    parser.add_argument('segmentation', default='', help='the location of speaker segmentation checkpoint.')
    parser.add_argument('embedding', default='', help='the location of speaker embedding checkpoint.')
    args = parser.parse_args()

    # unique code for the task
    code = generate_ramdom_code()   

    # load data
    protocol = load_data(args.corpus_name)
    # build the pipeline
    seg = PyanNet.load_from_checkpoint(args.segmentation)
    emb = XVectorMFCC.load_from_checkpoint(args.embedding)
    pipeline = SpeakerDiarizationPipeline(segmentation=seg, embedding=emb)
    hyper_params = tune_pipeline(pipeline, protocol, freeze_set={'min_duration_on': 0.0, 'min_duration_off': 0.0})
    # test performance
    test_performance(pipeline, protocol, calculate_diarization_error_rate, code)

    # raise success information
    print('The task is finished!')
