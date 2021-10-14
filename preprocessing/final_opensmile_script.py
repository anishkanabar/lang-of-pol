import os
import time

import numpy as np
import pandas as pd

import audb
import audiofile
import opensmile

import pydub
from sklearn import preprocessing
from pydub.silence import detect_nonsilent
from pydub import AudioSegment
from multiprocessing import Pool
from collections import defaultdict
from timeit import default_timer as timer
import glob, os, pickle, sys
import numpy as np
import pandas as pd

import audb
import audiofile
#import opensmile
#import VoiceLab

num_processors = 28
base_path='/media/4tb/data/' # base_path='/project2/mbspencer/nih/data/' ## MODIFIED
#zone='Zone '+str(sys.argv[1])

# Start timer
#start = timer()
zone = 'Zone 1'
date = '2018_08_05'

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

### from Feature Extraction (Praat)
def get_mp3_file(mp3_file):
    return {mp3_file.split('/')[-1]:AudioSegment.from_mp3(mp3_file)}

def load_audio(files,num_processors=num_processors):
    files.sort()
    p = Pool(processes = num_processors)
    output = p.map(get_mp3_file,files)
    p.close()
    global audio_dict
    audio_dict = {k:v for i in output for k,v in i.items()}
    return audio_dict

def pydub_to_np(audio):
    return np.array(audio.get_array_of_samples(), dtype=np.float64).reshape((-1, audio.channels)).T / (1<<(8*audio.sample_width)), audio.frame_rate

def normalized_df(df):
    min_max = preprocessing.MinMaxScaler()
    print("minmax")
    scaled_df = min_max.fit_transform(df.values)
    final_df = pd.DataFrame(scaled_df, columns=smile.feature_names)
    return final_df

def sound_to_df(sound):
    s = time.time()
    samples, sample_rate = pydub_to_np(sound)
    processed = smile.process_signal(samples,sample_rate)
    if not os.path.exists(f'./{zone}/{folder}/processed/'):
        os.makedirs(f'./{zone}/{folder}/processed/')
        os.makedirs(f'./{zone}/{folder}/normalized/')
    processed.to_csv(f'./{zone}/{folder}/processed/{d[sound]}.csv', index = True)
    normalized = normalized_df(processed)
    normalized.to_csv(f'./{zone}/{folder}/normalized/{d[sound]}.csv', index = True)
    print("time to process a single sound: \n" + str(time.time()-s))
    return normalized

def make_dataframe_parallel(audio_dict, folder, num_processors=num_processors): #previously files was audio_dict
    d_keys = list(audio_dict.keys())
    d_vals = list(audio_dict.values())
    print(folder)
    global d
    d = dict(zip(d_vals, d_keys))
    s = time.time()
    p = Pool(processes = num_processors)
    processed = p.map(sound_to_df, d_vals)
    e = time.time()
    print("time to process: "+ str(e-s))

def date_folder_iter(base_path, zone):
    date_folders = os.listdir(os.path.join(base_path, zone))
    for f in ['2018_09_01', '2018_09_02' ,'2018_09_08', '2018_09_24' ,'2018_10_23', '2018_11_07','2018_11_09', '2018_11_11' , '2018_11_26', '2018_11_29', '2018_12_19', '2019_01_07' ,'2019_01_17', '2019_01_19' , '2019_01_20']:
        date_folders.remove(f)
    print(date_folders)
    global folder
    for folder in date_folders:
        audio_path = os.path.join(base_path, zone, folder)
        print(audio_path)
        files = glob.glob(audio_path+'/2*.mp3')
        audio_dict = load_audio(files)
        s = time.time()
        make_dataframe_parallel(audio_dict, folder) #normalized = #(audio_dict, folder)
        e = time.time()
        print("time to make df in parallel: \n" + str(time.time() -s))
zone = "Zone1"
date_folder_iter(base_path, zone)
#folder = '2018_11_29'
#audio_path = base_path + zone.replace(' ','') + '/' + date + '/'
#files = glob.glob(audio_path+'2*.mp3')
#audio_dict = load_audio(files)
#        print(vals)
##sound = audio_dict['201808042331-339616-27730.mp3']
##samples, sample_rate = pydub_to_np(sound)
#make_dataframe_parallel(audio_dict, '2018_11_29')
#
##smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
##processed = smile.process_signal(samples,sample_rate)
#
##from sklearn import preprocessing
##min_max = preprocessing.MinMaxScaler()
##scaled_df = min_max.fit_transform(processed.values)
##final_df = pd.DataFrame(scaled_df,columns=smile.feature_names)
##	
##final_df.to_csv('test_extraction_normalized.csv', index = False)
##with open('./smile.log', 'r') as fp:
##    log = fp.readlines()
##log
#
