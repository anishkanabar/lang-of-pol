"""
speaker embedding component
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

import argparse
import pytorch_lightning as pl
from pyannote.audio.tasks import SpeakerEmbedding
from pyannote.audio.models.embedding import XVectorMFCC
from utils import generate_ramdom_code
from data_format import *
from evaluation import *
from tunning import *
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    # parse the command line
    parser = argparse.ArgumentParser(description='Speaker Segmentation component of the framework.')
    parser.add_argument('corpus_name', default='ami', help='the name of the corpus (default: AMI)')
    parser.add_argument('gpus', default=0, help='number of gpus.')
    parser.add_argument('pretrained', default=0, help='load pretrained models or not.')
    args = parser.parse_args()

    # unique code for the task
    code = generate_ramdom_code() 

    # load data
    protocol = load_data(args.corpus_name)
    # build the model
    emb = SpeakerEmbedding(protocol, duration=2.)
    model = XVectorMFCC(task=emb)
    # train the model, if gpu number is specified, then set the number of gpu, otherwise use cpu
    if args.gpus == 0:
        trainer = pl.Trainer(accelerator="cpu", max_time={"days": 2}, max_epochs=100,
                            default_root_dir=f'/Users/shiyang/Desktop/NIH/git/asr/results/shiyang_test/emb/')
    else:
        trainer = pl.Trainer(gpus=args.gpus, max_time={"days": 2}, max_epochs=100,
                            default_root_dir=f'/Users/shiyang/Desktop/NIH/git/asr/results/shiyang_test/emb/')
    trainer.fit(model)

    # raise success information
    print('The task is finished!')    


