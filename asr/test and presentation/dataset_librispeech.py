'''
File: dataset_librispeech.py
Brief: Loaders for librispeech transcripts and audio.
'''

import os
import pandas as pd

def describe(data, name):
    """
    Prints helpful statistics about dataset.
    Params
        @data: Fully loaded transcripts dataframe
    """
    print(f"{name} dataset stats:")
    print(f"\tRow count = {data.shape[0]}")


def load_transcripts(filepath, audio_type='.flac'):
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

