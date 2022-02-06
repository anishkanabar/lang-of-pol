"""
voice activity detection component
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

import argparse
import pytorch_lightning as pl
from pyannote.audio import Pipeline
from pyannote.audio.tasks import VoiceActivityDetection
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.pipelines import VoiceActivityDetection as VoiceActivityDetectionPipeline
from .utils import *
from .data_format import *
from .evaluation import *
from .tunning import *
import warnings
warnings.filterwarnings('ignore')


 
if __name__ == "__main__":
    # parse the command line
    parser = argparse.ArgumentParser(description='Voice Acitity Detection component of the framework.')
    parser.add_argument('corpus_name', default='ami', help='the name of the corpus (default: AMI)')
    parser.add_argument('gpus', default=0, help='number of gpus.')
    args = parser.parse_args()

    # if corpus name is ami, then load pre-trained pipeline from pyannote directly
    # otherwise, using data_format to formulate the data structure first
    if args.corpus_name == 'ami':
        # voice activity pipeline trained on AMI corpus
        pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
        protocol = load_data()
        hyper_params = tune_pipeline(pipeline, protocol, freeze_set={'min_duration_on': 0.0, 'min_duration_off': 0.0})
        test_performance(pipeline, protocol, calculate_detection_error_rate)
    else:
        # load data
        protocol = load_data(args.corpus_name)
        # build the model
        vad = VoiceActivityDetection(protocol, duration=2., batch_size=128)
        model = PyanNet(sincnet={'stride': 10}, task=vad)
        # train the model, if gpu number is specified, then set the number of gpu, otherwise use cpu
        if args.gpus == 0:
            trainer = pl.Trainer(accelerator="cpu", max_time={"days": 1})
        else:
            trainer = pl.Trainer(gpus=args.gpus, max_time={"days": 1})
        trainer.fit(model)
        # tune hyper-parameters
        pipeline = VoiceActivityDetectionPipeline(segmentation=model)
        hyper_params = tune_pipeline(pipeline, protocol, freeze_set={'min_duration_on': 0.0, 'min_duration_off': 0.0})
        # test performance
        test_performance(pipeline, protocol, calculate_diarization_error_rate)
    
    # raise success information
    print('The task is finished!')

