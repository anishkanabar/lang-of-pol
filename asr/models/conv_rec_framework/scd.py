"""
speaker segmentation component
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

import argparse
import pytorch_lightning as pl
from pyannote.audio import Pipeline
from pyannote.audio.tasks import Segmentation
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.pipelines import SpeakerSegmentation as SpeakerSegmentationPipeline
from utils import generate_ramdom_code
from data_format import *
from evaluation import *
from tunning import *
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    # parse the command line
    parser = argparse.ArgumentParser(description='Speaker Change Detection component of the framework.')
    parser.add_argument('corpus_name', default='ami', help='the name of the corpus (default: AMI)')
    parser.add_argument('gpus', default=0, help='number of gpus.')
    parser.add_argument('pretrained', default=0, help='load pretrained models or not.')
    args = parser.parse_args()

    # unique code for the task
    code = generate_ramdom_code()

    # if corpus name is ami, then load pre-trained pipeline from pyannote directly
    # otherwise, using data_format to formulate the data structure first
    if args.corpus_name == 'ami' and args.pretrained == 1:
        # speaker segmentation pipeline trained on AMI corpus
        pipeline = Pipeline.from_pretrained("pyannote/speaker-segmentation")
        protocol = load_data()
        hyper_params = tune_pipeline(pipeline, protocol, freeze_set={'min_duration_on': 0.0, 'min_duration_off': 0.0})
        test_performance(pipeline, protocol, calculate_detection_error_rate)
    else:
        # load data
        protocol = load_data()
        # build the model
        scd = Segmentation(protocol, duration=2., batch_size=128)
        model = PyanNet(sincnet={'stride': 10}, task=scd)
        # train the model, if gpu number is specified, then set the number of gpu, otherwise use cpu
        if args.gpus == 0:
            trainer = pl.Trainer(accelerator="cpu", max_time={"days": 2}, max_epochs=100,
                                default_root_dir=f'/Users/shiyang/Desktop/NIH/git/asr/results/shiyang_test/scd/')
        else:
            trainer = pl.Trainer(gpus=args.gpus, max_time={"days": 2}, max_epochs=100,
                                default_root_dir=f'/Users/shiyang/Desktop/NIH/git/asr/results/shiyang_test/scd/')
        trainer.fit(model)
        # tune hyper-parameters
        pipeline = SpeakerSegmentationPipeline(segmentation=model)
        hyper_params = tune_pipeline(pipeline, protocol, freeze_set={'min_duration_on': 0.0, 'min_duration_off': 0.0})
        # test performance
        test_performance(pipeline, protocol, calculate_segmentation_precision, code)
    
    # raise success information
    print('The task is finished!')
        