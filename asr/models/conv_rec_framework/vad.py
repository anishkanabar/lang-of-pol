"""
The framework of conversation recognition
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

from unittest import skip
import torch
import argparse
from pyannote.audio import Pipeline
from huggingface_hub import HfApi
from pyannote.database.util import load_rttm
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.core import Segment, notebook, SlidingWindowFeature
from pyannote.database import get_protocol, FileFinder
import warnings
warnings.filterwarnings('ignore')


def load_test(corpus='AMI'):
    """
    load test data
    @corpus: the name of corpus 
    """
    preprocessors = {'audio': FileFinder()}
    protocol = get_protocol(f"{corpus}.SpeakerDiarization.MixHeadset", preprocessors=preprocessors)
    return protocol.test()


def calculate_der(test_predition, test_files):
    """
    calculate the detection error rate
    @test_prediction: the prediction given by vad model based on test files
    @test_files: the reference files
    """
    metric = DetectionErrorRate(collar=0.0, skip_overlap=False)
    return sum([metric(reference, hypothesis) for reference, hypothesis in zip(test_predition, test_files)]) / len(test_predition)


def test_performance(files):
    """
    test the performance of the model
    @files: test files
    """
    test_prediction = []
    for test_file in files:
        vad = pipeline(test_file['audio'])
        test_prediction.append(vad)
    # calculate the performance
    print(f"The detection error rate is {calculate_der(test_prediction, files)}")


if __name__ == "__main__":
    # parse the command line
    parser = argparse.ArgumentParser(description='Voice Acitity Detection component of the framework.')
    parser.add_argument('corpus_name', default='ami', help='the name of the corpus (default: AMI)')
    args = parser.parse_args()

    # if corpus name is ami, then load pre-trained pipeline from pyannote directly
    # otherwise, using data_format to formulate the data structure first
    if args.corpus_name == 'ami':
        # voice activity pipeline trained on AMI corpus
        pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
        test_files = load_test()
        test_performance(test_files)
    else:
        pass


