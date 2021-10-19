#!/usr/bin/env python
# coding: utf-8

from deepspeech import Model
import pandas as pd
import numpy as np
from datetime import datetime
import pydub
import pickle
import os, sys

today = datetime.today().strftime('%Y-%m-%d')
today = today.replace('-','_')
today

os.chdir('/project/graziul/')

#beam = 250
#alpha = 0.3
#beta = 0.05

beam = int(sys.argv[1])
alpha = float(sys.argv[2])
beta = float(sys.argv[3])
batch_file = sys.argv[4]
files_to_process = pickle.load(open(batch_file,'rb'))

try:
    batch_num = batch_file.split('utt_batch')[1].split('.')[0].split('_')[-1]
    decode_by_utt = True
except:
    batch_num = batch_file.split('batch')[1].split('.')[0].split('_')[-1]
    decode_by_utt = False

def get_filenames(df):
    try:
        df['file'] = df_transcripts['file'].str.split('(\d*-\d*-\d*)',expand=True)[1]+'.mp3'
        return df
    except:
        print('Error finding "file" variable in DataFrame')

def get_date(df, keep_date_info=False):
    try:
        df['date'] = df['year'].astype(str)+'_'+df['month'].astype(str).apply(lambda x: x.zfill(2))+'_'+df['day'].astype(str).apply(lambda x: x.zfill(2))
        if keep_date_info:
            return df
        else:
            return df[[i for i in df.columns.values if i not in ['year','month','day']]].copy()
    except:
        print('Error processing "year", "month", and/or "day" variables in DataFrame')

def dt_to_seconds(df):
    try:
        for i in ['start_dt', 'end_dt']:
            var_name = i.split('_')[0]
            df[var_name] = (df[i]-np.datetime64('1900-01-01T00:00:00.000000000')).dt.total_seconds()
        return df
    except:
        print('Error processing "start_dt" and/or "end_dt" variables in DataFrame')
        
def get_asr_vars(df):
    df = dt_to_seconds(df) 
    return df[['zone','date','time','file','start','end','transcriber','transcription']].copy()

model_path = 'code/deepspeech/deepspeech-0.9.3-models.pbmm'
scorer_path = 'code/deepspeech/deepspeech-0.9.3-models.scorer'

ds = Model(model_path) 
ds.enableExternalScorer(scorer_path)
ds.setBeamWidth(beam)
ds.setScorerAlphaBeta(alpha,beta)

# Function to decode audio, optionally return 5 transcripts 
def decode_audio(data, meta=False):
    if meta:
        text = ds.sttWithMetadata(data, num_results=5)        
    else:
        text = ds.stt(data)
    return text

# Function to decode file, either as a whole or by each utterance identified
def ds_decode_file(file_metadata, pad=200, utt_list=None):
    # Load metadata
    zone, date, file = file_metadata
    file_name = '/'.join([zone,date,file])
    file_path = 'data/'+file_name
    # Load .mp3 file
    audio = pydub.AudioSegment.from_mp3(file_path)
    audio_16khz = audio.set_frame_rate(16000)
    num_samples = len(audio_16khz)
    if utt_list is not None:
        # Decode utterances in .mp3 file
        text_dict = {}
        for utt in utt_list:
            start, end = utt
            start = max(int(start*1000)-pad,0)
            end = min(int(end*1000)+pad, num_samples) 
            audio_slice = audio_16khz[start:end]
            samples_array = audio_slice.get_array_of_samples()
            data = np.frombuffer(samples_array, np.int16)
            text = decode_audio(data)
            text_dict[utt] = text
        return text_dict
    else:
        audio = audio.get_array_of_samples()
        data = np.frombuffer(audio, np.int16)        
        text = decode_audio(data)
        return text

ds_transcript_dict = {}
ds_transcript_dict['meta'] = {'model':model_path,
        'scorer':scorer_path,
        'beam_width':beam,
        'alpha':alpha,
        'beta':beta,
        'audio_files':files_to_process}  

error_files = []

for file_info in files_to_process:
    try:
        if decode_by_utt:
            file_metadata, utt_list = file_info
            text_dict = ds_decode_file(file_metadata, utt_list)
            ds_transcript_dict[file_metadata[-1]] = text_dict
            print('Decoded',str(len(text_dict)),'utterances from',file_name)
        else:
            file_metadata = file_info
            text = ds_decode_file(file_metadata)
            ds_transcript_dict[file_metadata[-1]] = text
            print('Decoded',file_metadata)
    except:
        error_files.append(file_metadata)
        continue

ds_transcript_dict['meta']['error_file_info'] = error_files

if decode_by_utt:
    out_file = 'features/asr/ds093_transcript_dict'+str(batch_num)+'utt_'+today+'.pk'
else:
    out_file = 'features/asr/ds093_transcript_dict'+str(batch_num)+'_'+today+'.pk'
    
pickle.dump(ds_transcript_dict,open(out_file,'wb'))
