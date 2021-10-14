#!/usr/bin/env python
# coding: utf-8

import pydub
from multiprocessing import Pool
import opensmile
import pandas as pd
import glob
import pickle 
import os
import sys
from timeit import default_timer as timer

num_processors = 48
zone='Zone '+str(sys.argv[1])

audio_path = "/scratch/midway3/graziul/"

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)
#print(smile.feature_names)

def get_GeMAPSv01b(f, nonsilent_slices):
    audio = pydub.AudioSegment.from_mp3(audio_date_path+'/'+f )
    nonsilent_audio = pydub.AudioSegment.empty()
    for nonsilent_slice in nonsilent_slices:
        start, end = nonsilent_slice
        # Pad with 250ms on either side
        nonsilent_audio += audio[max(0,start-250):min(end+250,len(audio))]
    nonsilent_audio_samples = nonsilent_audio.get_array_of_samples()
    df = smile.process_signal(nonsilent_audio_samples, audio.frame_rate)
    df = df.copy().reset_index()
    df['file'] = f
    return df

date_path = audio_path+zone.replace(' ','')+'/'
dates = [i.split('/')[-1] for i in glob.glob(date_path+'*') if '20' in i]
dates.sort()

zone_start = timer()

for date in dates: 
    date_start = timer()
    audio_date_path = date_path+date+'/'
    parquet_file = audio_date_path+date+'gemapsv01b.parquet'
    #if os.path.isfile(parquet_file):
    #    continue
    if len(glob.glob(audio_date_path+'*.mp3'))>0:
        vad_dict = pickle.load(open(audio_date_path+date+'vad_dict.pkl','rb'))
        if vad_dict!={}:
            nonsilent_slices_dict = {k:vad_dict[k]['pydub'][-24]['nonsilent_slices'] for k in vad_dict}
            p = Pool(processes = num_processors)
            list_of_dfs = p.starmap(get_GeMAPSv01b,[(f,nonsilent_slices_dict[f]) for f in nonsilent_slices_dict])
            p.close()
            df = pd.concat(list_of_dfs)
            df = df.sort_values(['file','start'])
            df.to_parquet(parquet_file,engine='fastparquet')
            for f in glob.glob(audio_date_path+'*.mp3'):
                os.remove(f)
            date_end = timer()
            elapsed = round(date_end-date_start)
            print("Processing",zone,date,"took",elapsed,"seconds")
        
zone_end = timer()
zone_elapsed = round((zone_end-zone_start)/60)
print("Processing",zone,"took",str(zone_elapsed),"minutes")

    

# Stop timer
zone_end = timer()

zone_elapsed = round((zone_end-zone_start)/60)

print("Processing",zone,"took",str(zone_elapsed),"minutes")
