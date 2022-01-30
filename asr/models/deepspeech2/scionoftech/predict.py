import argparse
import logging
import deepasr as asr
from asr_dataset.radio import RadioDataset
from asr_dataset.librispeech import LibriSpeechDataset
import warnings
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model.')
parser.add_argument('--ntrain', type=int, required=True, help='Number of samples used to train.')
parser.add_argument('--npred', type=int, default=20, help='Number of predictions to run.')
parser.add_argument('--loglvl', type=str, default='INFO')
args = parser.parse_args()

logger = logging.getLogger('asr.pred')
logger.setLevel(args.loglvl)

logger.debug('Loading checkpoint...')
model = asr.pipeline.load(args.checkpoint)
logger.debug('Loading dataset...')
dataset = RadioDataset('rcc', nrow=args.ntrain + args.npred).data.tail(args.npred)

for i in range(args.npred):
    logger.debug(f'Predicting {i}...')
    sample = dataset.iloc[i]
    sample_path = sample['path']
    sample_transcript = sample['transcripts']
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sample_prediction = model.predict(sample_path)
            logger.info(f"Trial {i}: TRUE: {sample_transcript}")
            logger.info(f"Trial {i}: PRED: {sample_prediction}") 
    except ValueError e:
        logger.error(f'Error Predicting {i}:\n{e}')
