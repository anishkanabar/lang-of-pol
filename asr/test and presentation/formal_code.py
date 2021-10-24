import os
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
# the build-in deepasr model in local
import deepasr as asr
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# log document
import logging
from keras.callbacks import CSVLogger
log = "general.log"
logging.basicConfig(filename=log,level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
csv_logger = CSVLogger('model_log.csv', append=True, separator=';')

def get_audio_trans(filepath, audio_type='.flac'):
    """
    This function is to get audios and transcripts needed for training
    @filepath: the path of the dicteory
    """
    count, k, inp = 0, 0, []
    audio_name, audio_trans = [], []
    for dir1 in os.listdir(filepath):
        if dir1 == '.DS_Store': continue
        dir2_path = filepath + dir1 + '/'
        for dir2 in os.listdir(dir2_path):
            if dir2 == '.DS_Store': continue
            dir3_path = dir2_path + dir2 + '/'
            
            for audio in os.listdir(dir3_path):
                if audio.endswith('.txt'):
                    k += 1
                    trans_path = dir3_path + audio
                    with open(trans_path) as f:
                        line = f.readlines()
                        for item in line:
                            flac_path = dir3_path + item.split()[0] + audio_type
                            audio_name.append(flac_path)
                            
                            text = item.split()[1:]
                            text = ' '.join(text)
                            audio_trans.append(text)
    return pd.DataFrame({"path": audio_name, "transcripts": audio_trans})


# get some thing to train the model
audio_trans = get_audio_trans('/Users/shiyang/Desktop/NIH//audio data/LibriSpeech/train-clean-100/')
train_data = audio_trans[audio_trans['transcripts'].str.len() < 100]


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


pipeline = get_config(feature_type='fbank', multi_gpu=False)
history = pipeline.fit(train_dataset=train_data, batch_size=128, epochs=500, callbacks=[csv_logger])
project_path = '/Users/shiyang/Desktop/NIH/'
pipeline.save(project_path + 'checkpoints')
