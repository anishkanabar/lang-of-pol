"""
speaker diarization framework
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

import getpass
import argparse
import datetime
import pytorch_lightning as pl
from pyannote.audio import Pipeline, Model
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
from utils import generate_ramdom_code
from data_format import *
from evaluation import *
from tunning import *
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "/scratch/midway3/" + getpass.getuser() + "/conv_rec/dia/"

if __name__ == "__main__":
    # parse the command line
    parser = argparse.ArgumentParser(description='Speaker diarization pipeline of the framework.')
    parser.add_argument('corpus_name', help='the name of the corpus.')
    parser.add_argument('sub_corpus', type=str, help='the name of the subcorpus.')
    parser.add_argument('-g', '--gpus', type=int, default=0, help='number of gpus.')
    parser.add_argument('-c', '--cpus', type=int, default=1, help='number of cpus.')
    parser.add_argument('-p', '--pretrained', type=str, default='n', help='whether use pretrained model.')
    parser.add_argument('-s', '--segmentation', type=str, default='n', help='load pretrained segementation model from path.')
    args = parser.parse_args()

    # unique code for the task
    code = generate_ramdom_code()
    print(f'DIA job {code}')
    print(f'Training Time: {datetime.datetime.now()}')   

    # create output folder
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if args.corpus_name == 'ami' and args.pretrained in ['y', 'Y', '1', 't', 'T']:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        protocol = load_data()
        if args.tunning in ['y', 'Y', '1', 't', 'T']:
            hyper_params = tune_pipeline(pipeline, protocol, freeze_set={'min_duration_on': 0.0, 'min_duration_off': 0.0})
            print("tunned successfully!")
            pipeline = pipeline.instantiate(hyper_params)
        test_performance(pipeline, protocol, calculate_diarization_error_rate, code)
    else:
        # load data
        protocol = load_data(args.corpus_name, args.sub_corpus)
        # load checkpoints of pre-trained segmentation from folders
        if len(os.listdir("/scratch/midway3/" + getpass.getuser() +'/conv_rec/scd/lightning_logs')) != 0:
            seg = Model.from_pretrained(args.segmentation)
        else:
            raise ValueError
        # load embedding model
        emb = Model.from_pretrained('pyannote/embedding')
        pipeline = SpeakerDiarizationPipeline(segmentation=seg, embedding=emb)
        hyper_params = tune_pipeline(pipeline, protocol, freeze_set={'min_duration_on': 0.0, 'min_duration_off': 0.0})
        pipeline = pipeline.instantiate(hyper_params)
        print("tunned successfully!")
        # test performance
        test_performance(pipeline, protocol, calculate_diarization_error_rate, code)

    # raise success information
    print('The task is finished!')
