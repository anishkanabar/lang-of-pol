import pprint
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from pydub import AudioSegment
import pickle, os, glob, numpy as np
from multiprocessing import Pool
base_path='/media/4tb/data/'
zone = 'Zone1'
write_path = "/home/ayahahmad/pyspeechfeat/"
num_processors = 28
date = "2018_09_01"

def get_mp3_file(mp3_file):
    return {mp3_file.split('/')[-1]:AudioSegment.from_mp3(mp3_file)}

def load_audio(files,num_processors=num_processors):
    files.sort()
    p = Pool(processes = 28)
    output = p.map(get_mp3_file,files)
    p.close()
    audio_dict = {k:v for i in output for k,v in i.items()}
    return audio_dict

def pydub_to_np(audio):
    return np.array(audio.get_array_of_samples(), dtype=np.float64).reshape((-1, audio.channels)).T / (1<<(8*audio.sample_width)), audio.frame_rate

###############################################################################################################################################################
to_something = None

def sound_to_something_array(sound):
    samples, sample_rate = pydub_to_np(sound)
    print(samples)
    if to_something == 'mfcc':
        something = mfcc(samples,sample_rate, nfilt = 40, nfft=600)
    elif to_something == 'delta':
        mfcc_feat = mfcc(samples,sample_rate, nfft=600)
        something = delta(mfcc_feat, 2)
    elif to_something == 'logfbank':
        something = logfbank(samples, sample_rate, nfft=600)
    return something

#def peak_normalization(array):
#    new_array = (array - array.min())/(array.max() - array.min())
#    return new_array

#def make_single(sound):
#    temp_array = sound_to_something_array(sound)
#    normalized_array = peak_normalization(temp_array)
#    print(normalized_array)
#    return normalized_array

def make_dict_parallel(audio_dict, folder, num_processors=num_processors):
    ''' does everything in parallel'''

    d_keys = list(audio_dict.keys())
    d_vals = list(audio_dict.values())
    
    print(d_vals)
    #p = Pool(processes = 8)
    result = map(sound_to_something_array, d_vals)
    print(result)
    dictionary = {}
    for k, r in zip(d_keys, result):
        dictionary[k] = r
    #p.close()

    #with open(f"./{folder}_{to_something_name}_dict.pkl", "wb") as fil:
    #    pickle.dump(dictionary, fil, pickle.HIGHEST_PROTOCOL)
    return dictionary

def date_folder_iter(base_path, zone, daily_dict):
    date_folders = os.listdir(os.path.join(base_path, zone))
    [date_folders.remove(i) for i in date_folders if not i.startswith('2')]
    for folder in date_folders:
        audio_path = os.path.join(base_path, zone, folder)
        files = glob.glob(audio_path+'/2*.mp3')
        audio_dict = load_audio(files)
        daily_dict[folder] = make_dict_parallel(audio_dict, folder)
    if not os.path.exists(f'/{write_path}/{zone}/{to_something}/'):
        os.makedirs(f'/{write_path}/{zone}/{to_something}/')
    with open(f"/{write_path}/{zone}/{to_something}/daily_{to_something}_dict.pkl", "wb") as fil:
        pickle.dump(daily_dict, fil, pickle.HIGHEST_PROTOCOL)
    return daily_dict



# test code
audio_path = base_path + zone.replace(' ','') + '/' + date + '/'
files = glob.glob(audio_path+'2*.mp3')
audio_dict = load_audio(files)
sound = audio_dict["201808312336-496425-27730.mp3"]

to_something = "delta"  #"mfcc"
mfcc_dict={}
#make_single(sound)
date_folder_iter(base_path, zone, mfcc_dict)
#a = sound_to_something_array(sound)
#print(a)
#samples, sample_rate = pydub_to_np(sound)
#mfcc_feat = mfcc(samples,sample_rate, nfft=600)
#d_mfcc_feat = delta(mfcc_feat, 2)
#fbank_feat = logfbank(samples, sample_rate, nfft=600)
#print(mfcc_feat)
#print(d_mfcc_feat)
#print(fbank_feat)
