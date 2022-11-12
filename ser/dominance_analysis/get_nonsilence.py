#!/usr/bin/env python
# coding: utf-8

import pydub
from pydub.silence import detect_nonsilent
from pydub import AudioSegment
from multiprocessing import Pool
from collections import defaultdict
from timeit import default_timer as timer
import glob, os, pickle, sys

num_processors = 28
base_path='/project2/mbspencer/nih/data/'
zone='Zone '+str(sys.argv[1])

# Start timer
start = timer()

# Create a dictionary {mp3_file:AudioSegment.from_mp3(mp3_file)}
def get_mp3_file(mp3_file):
    return {mp3_file.split('/')[-1]:AudioSegment.from_mp3(mp3_file)}
# Create a dictionary {mp3_file:[lists of start/stop for nonsilent portions of audio]}
def get_nonsilent_slices(file, audio, min_silence, silence_thresh):
    return {file:detect_nonsilent(audio,min_silence_len=min_silence,silence_thresh=silence_thresh)}

# Create a class holding all files/metadata for one day of data in one dispatch zone
class DailyGrind:

    # Initialize to include basic metadata, particularly for locating files
    def __init__(self, zone, date, base_path):
        self.zone = zone
        self.date = date
        self.base_path = base_path
        self.audio_path = base_path + zone.replace(' ','') + '/' + date + '/'
        self.files = glob.glob(self.audio_path+'2*.mp3')
        self.filenames = [i.split('/')[-1] for i in self.files]
        # At least 500ms of silence when detecting non-silence
        self.min_silence=500
        # Silence detection threshold (dB)
        self.thresh=-24
    
    # Function to load audio files 
    def load_audio(self, num_processors=num_processors):
        files = self.files
        files.sort()
        p = Pool(processes = num_processors)
        output = p.map(get_mp3_file,files)
        p.close()
        self.audio_dict = {k:v for i in output for k,v in i.items()}

    # Function to perform voice activity detection (VAD) 
    def VAD(self, method='pydub'):
        # Method: pydub 'detect_nonsilent' function (energy-based using dB threshold)
        if method=='pydub':
            p = Pool(processes = num_processors)
            output = p.starmap(get_nonsilent_slices,[(filename,self.audio_dict[filename],self.min_silence, self.thresh) for filename in self.audio_dict])
            p.close()
            self.nonsilent_slices_dict = {k:v for i in output for k,v in i.items()}

    # Function to extract metadata into a dictionary
    def get_metadata(self,method,overwrite=False):
        # Check if metadata dictionary already exists
        metadata_exists = os.path.isfile(self.base_path+self.zone.replace(' ','')+'/'+self.date+'/'+self.date+'metadata_dict.pkl')
        if ~metadata_exists:
            # Create metadata dictionary and populate with cursory metadata
            metadata_dict={}
            metadata_dict['zone'] = self.zone
            metadata_dict['date'] = self.date
            # Find files that are 100% silence
            metadata_dict['files_total_silence'] = [k for k in self.nonsilent_slices_dict if len(self.nonsilent_slices_dict[k])==0]
            # Tag if date has files with 100% silence
            metadata_dict['has_silent_files'] = len(metadata_dict['files_total_silence'])>0
            # Find duration of recordings (discounting files with 100% silence)
            metadata_dict['file_length_seconds'] = {filename:self.audio_dict[filename].duration_seconds for filename in self.audio_dict if filename not in metadata_dict[method]['files_total_silence']}
            # Find duration of entire day
            metadata_dict['day_length_minutes'] = sum(metadata_dict['file_length_seconds'].values())/60 
            # Day has 95%+ coverage?
            metadata_dict['complete_data'] = metadata_dict['day_length_minutes']/1440>=0.95
            return metadata_dict
        
    # Function to extract VAD metadata into a dictionary
    def get_vad_metadata(self,method='pydub',overwrite=False):
        # Check if VAD metadata dictionary exists
        vad_metadata_exists = os.path.isfile(self.base_path+self.zone.replace(' ','')+'/'+self.date+'/'+self.date+'vad_dict.pkl')
        if ~vad_metadata_exists:
            # Create VAD metadata dictionary
            vad_dict = {}
            # Create entry for silence detection method
            vad_metadata_dict[method] = {}
            # Extract timestamp metadata and add timing of non-silence
            for filename in self.nonsilent_slices_dict:
                timestamp = filename.split('.')[0]
                vad_metadata_dict[method][filename] = {}
                vad_metadata_dict[method][filename]['recording_start'] = {'year':int(timestamp[0:4]),
                                                                          'month':int(timestamp[4:6]),
                                                                          'day':int(timestamp[6:8]),
                                                                          'time':int(timestamp[8:12])}
                # Get nonsilent slices from before
                file_nonsilent_slices = self.nonsilent_slices_dict[filename]
                # Save them for later use
                vad_metadata_dict[method][filename]['nonsilent_slices'] = file_nonsilent_slices
                # Load audio for the file
                audio = self.audio_dict[filename]
                # Set number of nonsilent seconds to 0
                nonsilent_seconds = 0
                # If any nonsilent slices...
                if len(file_nonsilent_slices)>0:
                    # For each nonsilent slice:
                    for nonsilent_slice in file_nonsilent_slices:
                        # Extract start/end of nonsilent slice
                        start, end = nonsilent_slice
                        # Get duration of non-silence before adding buffer of silence between slices
                        nonsilent_seconds += audio[start:end].duration_seconds
                vad_metadata_dict[method][filename]['nonsilent_minutes'] = nonsilent_seconds/60
            if method=='pydub':
                # Include threshold used for pydub silence detection
                vad_metadata_dict[method]['thresh'] = self.thresh
            return vad_metadata_dict

def get_metadata_dict(zone, date, base_path):
    print('Processing '+date+' in '+zone)
    dg = DailyGrind(zone=zone,date=date,base_path=base_path)
    dg.load_audio()
    dg.VAD()
    metadata_dict = dg.get_metadata(method='pydub')
    pickle.dump(metadata_dict,open(base_path+zone.replace(' ','')+'/'+date+'/'+date+'metadata_dict.pkl','wb'))    

date_path = base_path+zone.replace(' ','')+'/'
dates = [i.split('/')[-1] for i in glob.glob(date_path+'*') if '20' in i]
dates.sort()
for date in dates:
    get_metadata_dict(zone, date, base_path)

# Stop timer
end = timer()

elapsed = (end-start)/60

print("Completed "+zone+" in "+str(elapsed)+" minutes")

