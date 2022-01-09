'''
File: atczero.py
Brief: Loader for ATC0 corpus.
Authors: Eric Chandler <echandler@uchicago.edu>
'''

import os
import re
import datetime
import pandas as pd
from dataset import AudioClipDataset

class ATCZeroDataset(AudioClipDataset):

    @classmethod
    def load_transcripts(cls, transcripts_dir, sample_rate=SAMPLE_RATE):
        """
        This function is to get audios and transcripts needed for training
        Params:
            @transcripts_dir: path to directory with transcripts csvs
        """
        pass
 
