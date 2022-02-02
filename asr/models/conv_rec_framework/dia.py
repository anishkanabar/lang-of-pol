"""
speaker diarization framework
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

import argparse
from pyannote.audio import Pipeline
from .utils import *
from .data_format import *
from .evaluation import *
from .tunning import *
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    # parse the command line
    parser = argparse.ArgumentParser(description='Speaker diarization pipeline of the framework.')
    parser.add_argument('corpus_name', default='ami', help='the name of the corpus (default: AMI)')
    args = parser.parse_args()

    # if corpus name is ami, then load pre-trained pipeline from pyannote directly
    # otherwise, using data_format to formulate the data structure first
    if args.corpus_name == 'ami':
        # spekaer recognition pipeline trained on AMI corpus
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        protocol = load_data()
        hyper_params = tune_pipeline(pipeline, protocol, freeze_set={'min_duration_on': 0.0, 'min_duration_off': 0.0})
        test_performance(pipeline, protocol, calculate_diarization_error_rate)
    else:
        pass


