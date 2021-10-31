import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# log document
import logging
from keras.callbacks import CSVLogger

import deepasr as asr
import dataset_librispeech as libri_data
import dataset_radio as radio_data

def configure_logging(log_dir):
    """
    Create loggers to record training progress.
    @log_dir: Directory to write logs
    """
    logging.basicConfig(filename=os.path.join(log_dir, 'general.log'),
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s', 
                        datefmt='%d/%m/%Y %H:%M:%S')
    csv_logger =  CSVLogger(filename=os.path.join(log_dir, 'model_log.csv'),
                            append=True, 
                            separator=';')
    return csv_logger


def get_config(feature_type = 'spectrogram', multi_gpu = False):
    """
    Get the CTC pipeline
    @feature_type: the format of our dataset
    @multi_gpu: whether using multiple GPU
    """
    # audio feature extractor, this is build on asr built-in methods
    features_extractor = asr.features.preprocess(feature_type=feature_type, features_num=161,
                                                 samplerate=16000,
                                                 winlen=0.02,
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
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-2,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-4
    )
    
    # output label deocder
    decoder = asr.decoder.GreedyDecoder()
    
    # CTC Pipeline
    pipeline = asr.pipeline.ctc_pipeline.CTCPipeline(
        alphabet=alphabet_en, features_extractor=features_extractor, model=model, optimizer=optimizer, decoder=decoder,
        sample_rate=16000, mono=True, multi_gpu=multi_gpu
    )
    return pipeline


if __name__ == "__main__":
    #project_path = '/project/graziul/ra/shiyanglai/experiment1'
    project_path = '/project/graziul/ra/echandler/experiment1'
    libri_dir = '/project/graziul/ra/shiyanglai/experiment1/audio data/LibriSpeech/train-clean-100/'
    #audio_trans = libri_data.load_transcripts(libri_dir)
    audio_trans = radio_data.load_transcripts('/project/graziul/transcripts')
    train_data = audio_trans[audio_trans['transcripts'].str.len() < 100]
    train_data = train_data.head()
    print(train_data[['offset','duration']])
    #csv_logger = configure_logging(project_path)
    #pipeline = get_config(feature_type='fbank', multi_gpu=False)
    #history = pipeline.fit(train_dataset=train_data, batch_size=128, epochs=500, callbacks=[csv_logger])
    #history = pipeline.fit(train_dataset=train_data, batch_size=64, epochs=10, callbacks=[csv_logger])
    #pipeline.save(os.path.join(project_path, 'checkpoints'))
