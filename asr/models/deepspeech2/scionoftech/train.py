"""
File: train_deepspeech.py
Brief: Trains deepspeech 2 model on a dataset
Usage: python -m train_deepspeech <out_dir>
"""
import os
import logging
import argparse
import pathlib
import datetime as dt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from keras.callbacks import CSVLogger, ModelCheckpoint
import tensorflow as tf
import deepasr as asr
from asr_dataset.librispeech import LibriSpeechDataset
from asr_dataset.radio import RadioDataset

SAMPLE_RATE = 16000   # Hz
WINDOW_LEN = .02 # Sec
NUM_TRAIN = 8192 #16384

app_logger = logging.getLogger('main.train')

def configure_logging(log_dir):
    """
    Create loggers to record training progress.
    @log_dir: Directory to write logs
    """
    logging.basicConfig(filename=os.path.join(log_dir, 'general.log'),
                        level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s', 
                        datefmt='%d/%m/%Y %H:%M:%S')
    return CSVLogger(filename=os.path.join(log_dir, 'model_log.csv'),
                            append=True, 
                            separator=';')


def define_model(feature_type = 'spectrogram', multi_gpu = False):
    """
    Get the CTC pipeline
    @feature_type: the format of our dataset
    @multi_gpu: whether using multiple GPU
    """
    # audio feature extractor, this is build on asr built-in methods
    features_extractor = asr.features.preprocess(feature_type=feature_type, 
                                                 features_num=161,
                                                 samplerate=SAMPLE_RATE,
                                                 winlen=WINDOW_LEN,
                                                 winstep=0.025,
                                                 winfunc=np.hanning)

    # input label encoder
    alphabet_en = asr.vocab.Alphabet(lang='en')
    
    # training model
    model = asr.model.get_deepasrnetwork1(
        input_dim=161,
        output_dim=29,
        is_mixed_precision=True
    )
    
    # model optimizer
    optimizer = 'RMSprop'
    
    # output label deocder
    decoder = asr.decoder.GreedyDecoder()
    
    # CTC Pipeline
    pipeline = asr.pipeline.ctc_pipeline.CTCPipeline(
        alphabet=alphabet_en,
        features_extractor=features_extractor, 
        model=model, 
        optimizer=optimizer, 
        decoder=decoder,
        sample_rate=SAMPLE_RATE, 
        mono=True, 
        multi_gpu=multi_gpu
    )

    return pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['librispeech', 'radio'])
    parser.add_argument('cluster', choices=['rcc', 'ai'])
    parser.add_argument('output_dir', type=pathlib.Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.join(args.output_dir, 'job_' + os.environ.get('SLURM_JOB_ID','0'))
    os.makedirs(output_dir, exist_ok=True)
    model_logger = configure_logging(output_dir)

    tick = dt.datetime.now()
    if args.dataset == 'librispeech':
        dataset_loader = LibriSpeechDataset(args.cluster, nrow=NUM_TRAIN, window_len=WINDOW_LEN)
    else:
        dataset_loader = RadioDataset(args.cluster, nrow=NUM_TRAIN, window_len=WINDOW_LEN)
    app_logger.info("Dataset load success.")

    pipeline = define_model(feature_type='spectrogram', multi_gpu=True)

    epoch_checkpoint_dir = os.path.join(output_dir, 'epoch_checkpoints') 
    os.makedirs(epoch_checkpoint_dir, exist_ok=True)
    # TODO: Monitor loss instead of accuracy once we start getting non-infinite loss
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(epoch_checkpoint_dir, 'checkpoint-epoch-{epoch:02d}.hdf5'),
        save_weights_only=False,
        save_freq='epoch',
        monitor='accuracy',
        mode='max',
        save_best_only=True)

    app_logger.info("Pipeline model configured.")

    history = pipeline.fit(train_dataset=dataset_loader.data,
                           batch_size=64, 
                           epochs=500, 
                           callbacks=[model_logger, model_checkpoint])
    app_logger.info("Finished training.")
    tock = dt.datetime.now()
    app_logger.info(f"Elapsed: {tock - tick}")

    pipeline.save(os.path.join(output_dir, 'final_checkpoint'))
    app_logger.info("Model save success.")
