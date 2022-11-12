"""
speaker embedding component
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

import os
import datetime
import getpass
import argparse
import pytorch_lightning as pl
from pyannote.audio import Pipeline, Model
from pyannote.audio.tasks import SpeakerEmbedding
from pyannote.audio.models.embedding.debug import SimpleEmbeddingModel
from utils import generate_ramdom_code
from data_format import *
from evaluation import *
from tunning import *
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "/scratch/midway3/" + getpass.getuser() + "/conv_rec/emb/"

if __name__ == "__main__":
    # parse the command line
    parser = argparse.ArgumentParser(description='Speaker Embedding component of the framework.')
    parser.add_argument('corpus_name', help='the name of the corpus.')
    parser.add_argument('sub_corpus', type=str, help='the name of the subcorpus.')
    # parser.add_argument('tunning', type=str, help='tune hyperparameters or not.')
    parser.add_argument('-g', '--gpus', type=int, default=0, help='number of gpus.')
    parser.add_argument('-c', '--cpus', type=int, default=1, help='number of cpus.')
    parser.add_argument('-p', '--pretrained', type=str, default='n', help='load pretrained models from path.')
    parser.add_argument('-e', '--epoch', type=int, default=0, help='number of epochs for training.')
    args = parser.parse_args()

    # unique code for the task
    code = generate_ramdom_code()
    print(f'EMB job {code}')
    print(f'Training Time: {datetime.datetime.now()}')

    # create output folder
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # load data
    if args.corpus_name == 'ami' and args.pretrained in ['y', 'Y', '1', 't', 'T']:
        pipeline = Pipeline.from_pretrained("pyannote/")
    # build the model
    emb = SpeakerEmbedding(protocol, duration=2.)
    model = XVectorMFCC(task=emb)
    # train the model, if gpu number is specified, then set the number of gpu, otherwise use cpu
    if args.gpus == 0:
        trainer = pl.Trainer(accelerator="cpu", max_time={"days": 2}, max_epochs=100,
                            default_root_dir=f'results/conv_rec/emb/')
    else:
        trainer = pl.Trainer(gpus=args.gpus, max_time={"days": 2}, max_epochs=100,
                            default_root_dir=f'results/conv_rec/emb/')
    trainer.fit(model)

    # raise success information
    print('The task is finished!')    


