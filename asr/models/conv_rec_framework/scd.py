"""
speaker segmentation component
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

import os
import datetime
import getpass
import argparse
import pytorch_lightning as pl
from pyannote.audio import Pipeline, Model
from pyannote.audio.tasks import Segmentation
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.pipelines import SpeakerSegmentation as SpeakerSegmentationPipeline
from utils import generate_ramdom_code
from data_format import *
from evaluation import *
from tunning import *
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "/scratch/midway3/" + getpass.getuser() + "/conv_rec/scd/"

if __name__ == "__main__":
    # parse the command line
    parser = argparse.ArgumentParser(description='Speaker Change Detection component of the framework.')
    parser.add_argument('corpus_name', help='the name of the corpus.')
    parser.add_argument('sub_corpus', type=str, help='the name of the subcorpus.')
    parser.add_argument('tunning', type=str, help='tune hyperparameters or not.')
    parser.add_argument('-g', '--gpus', type=int, default=0, help='number of gpus.')
    parser.add_argument('-c', '--cpus', type=int, default=1, help='number of cpus.')
    parser.add_argument('-p', '--pretrained', type=str, default='n', help='load pretrained models from path.')
    parser.add_argument('-e', '--epoch', type=int, default=0, help='number of epochs for training.')
    args = parser.parse_args()

    # unique code for the task
    code = generate_ramdom_code()
    print(f'SCD job {code}')
    print(f'Training Time: {datetime.datetime.now()}')

    # create output folder
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # if corpus name is ami, then load pre-trained pipeline from pyannote directly
    # otherwise, using data_format to formulate the data structure first
    if args.corpus_name == 'ami' and args.pretrained in ['y', 'Y', '1', 't', 'T', 'yes', 'Yes', 'YES']:
        # speaker segmentation pipeline trained on AMI corpus
        pipeline = Pipeline.from_pretrained("pyannote/speaker-segmentation")
        protocol = load_data()
        if args.tunning in ['y', 'Y', '1', 't', 'T', 'yes', 'Yes', 'YES']:
            hyper_params = tune_pipeline(pipeline, protocol, freeze_set={'min_duration_on': 0.0, 'min_duration_off': 0.0})
            print("tunned successfully!")
        test_performance(pipeline, protocol, calculate_detection_error_rate)
    else:
        # load data
        protocol = load_data(args.corpus_name, args.sub_corpus)
        # build the model
        if len(os.listdir(OUTPUT_DIR+'lightning_logs')) != 0 and args.pretrained != 'n':
            scd = Segmentation(protocol, duration=2., batch_size=128)
            model = PyanNet(sincnet={'stride': 10}, task=scd)
            if args.epoch != 0:
                # train the model, if gpu number is specified, then set the number of gpu, otherwise use cpu
                if args.gpus == 0:
                    trainer = pl.Trainer(num_processes=args.cpus, max_time={"minutes": 1}, max_epochs=args.epoch,
                                        accelerator="ddp", default_root_dir=OUTPUT_DIR)
                else:
                    trainer = pl.Trainer(gpus=args.gpus, max_time={"minutes": 1}, max_epochs=args.epoch,
                                        accelerator="ddp", default_root_dir=OUTPUT_DIR)
                trainer.fit(model, ckpt_path=args.pretrained)
        else:
            scd = Segmentation(protocol, duration=2., batch_size=128)
            model = PyanNet(sincnet={'stride': 10}, task=scd)
            # train the model, if gpu number is specified, then set the number of gpu, otherwise use cpu
            if args.epoch == 0:
                max_epoch = 20
            else:
                max_epoch = args.epoch
            # train the model, if gpu number is specified, then set the number of gpu, otherwise use cpu
            if args.gpus == 0:
                trainer = pl.Trainer(num_processes=args.cpus, max_time={"minutes": 1}, max_epochs=max_epoch,
                                    default_root_dir=OUTPUT_DIR)
            else:
                trainer = pl.Trainer(gpus=args.gpus, max_time={"minutes": 1}, max_epochs=max_epoch,
                                    default_root_dir=OUTPUT_DIR)
            trainer.fit(model)
        print("trained successfully!")
        # tune hyper-parameters
        if args.tunning in ['y', 'Y', '1', 't', 'T', 'yes', 'Yes', 'YES']:
            pipeline = SpeakerSegmentationPipeline(segmentation=model)
            hyper_params = tune_pipeline(pipeline, protocol, freeze_set={'min_duration_on': 0.0, 'min_duration_off': 0.0})
            print("tunned successfully!")
            pipeline = pipeline.instantiate(hyper_params)
        # test performance
        test_performance(pipeline, protocol, calculate_segmentation_precision, code)
    # raise success information
    print('The task is finished!')
        