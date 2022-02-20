import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchaudio
import torch
import numpy as np
import pandas as pd
import os
import pickle
import re
import torchaudio.transforms as T
import math
import librosa
import librosa.display
import matplotlib.patches as patches
from glob import glob

torch.manual_seed(1)

#Testing atc 0
def process_atc_label_line(line):
    lines_list = str.split(line, "\n")
    times_list = []
    for elt in lines_list:
        if elt[0:4] == " (TI":
            num_string = ['1','2','3','4','5','6','7','8','9','0',' ','.']
            new_elt = ''.join(c for c in elt if c in num_string)
            new_elt_tok = new_elt.split() 
            new_elt_tok = [float(num) for num in new_elt_tok]
            times_list.append(new_elt_tok)
    return times_list

class atc_audio_file():
    def __init__(self, name,audio_path, labels_path,new_flag = 0):
        self.name = name
        self.vad_slices = None
        self.frames = None
        self.frames_labels = None
        self.mfcc = None
        self.n_clips = 100
        self.flag = new_flag
        self.labels_path = labels_path
        self.audio_path = audio_path
    
    def get_slices(self):
        labels_path = self.labels_path
        with open(labels_path, 'r') as f:
            label = f.read()
        label = process_atc_label_line(label)
        self.vad_slices = label
        return self.vad_slices
    
    def get_frames(self):
        ms_2_sample = self.sample_rate/1000
        frames_array = np.zeros(self.mfcc.shape[2])

        for v in self.vad_slices:
            start = math.floor(v[0]*ms_2_sample)
            end = math.ceil(v[1]*ms_2_sample)
            #print(v)
            for i in range(start,end):
                n = math.floor(i/220)
                j = i%220
                if j <= 110:
                    frames_array[n-2] += 1
                    frames_array[n-1] += 1
                    frames_array[n] += 1
                elif j>=111 and j<=220:
                    frames_array[n-1] += 1
                    frames_array[n] += 1
                elif j>=221 and j<=330:
                    frames_array[n-1] += 1
                    frames_array[n] += 1
                    frames_array[n+1] += 1
                elif j>=331 and j<=440:
                    frames_array[n+1] += 1
                    frames_array[n] += 1
                elif j>=441:
                    frames_array[n+2] += 1
                    frames_array[n+1] += 1
                    frames_array[n] += 1
            
        self.frames = frames_array
        return self.frames
        
    def get_split_frames(self):
        ms_2_sample = self.sample_rate/1000
        frames_array = np.zeros(self.mfcc.shape[2]*self.n_clips)
        
        for v in self.vad_slices:
            start = math.floor(v[0]*ms_2_sample)
            end = math.ceil(v[1]*ms_2_sample)
            for i in range(start,end):
                n = min(math.floor(i/220),len(frames_array)-1)
                j = i%220
                if j <= 110:
                    frames_array[n-2] += 1
                    frames_array[n-1] += 1
                    frames_array[n] += 1
                elif j>=111 and j<=220:
                    frames_array[n-1] += 1
                    frames_array[n] += 1
                elif j>=221 and j<=330:
                    frames_array[n-1] += 1
                    frames_array[n] += 1
                    frames_array[n+1] += 1
                elif j>=331 and j<=440:
                    frames_array[n+1] += 1
                    frames_array[n] += 1
                elif j>=441:
                    frames_array[n+2] += 1
                    frames_array[n+1] += 1
                    frames_array[n] += 1
        
        self.clip_size = self.mfcc.shape[2]
        frame_arr_list = []
        for j in range(self.n_clips):
            frame_arr_list.append(np.expand_dims(frames_array[j*self.clip_size:(j+1)*self.clip_size],axis=0))
        self.frames = np.concatenate(frame_arr_list,axis=0)
        return self.frames
    
        
    def get_labels(self): 
        self.frames_labels = np.zeros(len(self.frames))
        self.frames_labels[np.where(self.frames>0)] = 1
        return self.frames_labels
    
    def get_split_labels(self):
        self.frames_labels = np.zeros_like(self.frames)
        self.frames_labels[np.where(self.frames>0)] = 1
        return self.frames_labels
        
    def get_mfcc(self): 
        if self.flag == 0:
            file_name = self.audio_path
        else:
            file_name = self.name
        self.waveform, self.sample_rate = torchaudio.load(file_name)
        pad_array = np.zeros([1,180000000*self.sample_rate])
        pad_array[:,self.waveform.shape[1]] = self.waveform
        self.waveform = pad_array
        self.waveform = self.waveform[:,:170000000*self.sample_rate] 
        n_fft = 2048
        win_length = 551
        hop_length = 220
        n_mels = 40
        n_mfcc = 40

        mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
              'n_fft': n_fft,
              'n_mels': n_mels,
              'hop_length': hop_length,
              'mel_scale': 'htk',
            }
        )

        self.mfcc = mfcc_transform(self.waveform)
        return self.mfcc
    
    def get_split_mfcc(self):
        if self.flag == 1:
            file_name = self.name
        else:
            file_name = self.audio_path
        waveform, sample_rate = torchaudio.load(file_name)
        effects = [['rate', '22050']]
        self.waveform, self.sample_rate = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
        pad_array = torch.zeros((1,10000*self.sample_rate))
        pad_array[:,:self.waveform.shape[1]] = self.waveform
        self.waveform = pad_array
        self.waveform = self.waveform[:,:4000*self.sample_rate]
        clip_size = math.floor(self.waveform.shape[1]/self.n_clips)
        n_clips = self.n_clips
        mfcc_list = []
        n_fft = 2048
        win_length = 551
        hop_length = 220
        n_mels = 40
        n_mfcc = 40
        mfcc_transform = T.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                  'n_fft': n_fft,
                  'n_mels': n_mels,
                  'hop_length': hop_length,
                  'mel_scale': 'htk',
                }
            )
        for i in range(n_clips):
            mfcc_list.append(mfcc_transform(self.waveform[:,i*clip_size:(i+1)*clip_size]))
        self.mfcc = torch.cat(mfcc_list)
        return self.mfcc
    
    def plot_waveform_with_labels(self,i,clip_size):
        plt.figure(figsize=(14,5))
        fig,(ax1,ax2) = plt.subplots(2,1)
        librosa.display.waveshow(self.waveform.squeeze().numpy()[i*clip_size:(i+1)*clip_size],self.sample_rate,ax = ax1)
        ax2.plot(self.frames_labels[i])
        plt.show()
        return    
    
    def get_plots(self): 
        clip_size = math.floor(162019000*self.sample_rate/self.n_clips)
        for i in range(self.n_clips):
            print(i)
            self.plot_waveform_with_labels(i,clip_size)
        return
    
def process_file_atc0(filename,audio_path, labels_path):
    fname = filename
    audio_file = atc_audio_file(fname, audio_path, labels_path)
    audio_file.get_slices()
    mfcc = audio_file.get_split_mfcc()
    frames = audio_file.get_split_frames()
    print(filename)
    return mfcc, frames

def process_atc0_files():
    input_list = []
    labels_list = []
    #paths = ['/project/graziul/data/corpora/atc0_comp/atc0_bos/data/audio/', '/project/graziul/data/corpora/atc0_comp/atc0_dca/data/audio/', '/project/graziul/data/corpora/atc0_comp/atc0_dfw/data/audio/']
    paths = ['/project/graziul/data/corpora/atc0_comp/atc0_bos/data/audio/']
    for idx,path in enumerate(paths):
        for fpath in glob(path + '*.sph'):
            if(idx > 6):
                break
            filename = fpath[-12:-4]
            label_file = path[:-6] + 'transcripts/' + filename + '.txt'
            x,y = process_file_atc0(filename, fpath, label_file)
            input_list.append(x)
            labels_list.append(y)
            idx = idx+1
        if(idx>6):
            break
    input_list = torch.cat(input_list)
    input_list = torch.transpose(input_list,1,2)
    labels_list = torch.from_numpy(np.concatenate(labels_list,axis = 0)).float()
    return input_list, labels_list

def load_data(pkl_path = '/project/graziul/ra/ajays/whitelisted_vad_dict.pkl'):
    file = open(pkl_path,'rb')
    vad_dict = pickle.load(file)
    file.close()
    input_list = []
    labels_list = []

    for idx,key in enumerate(vad_dict):
        print(idx)
        if idx > 100000000:
            break
        a = audio_file(key)
        a.get_slices(vad_dict)
        input_list.append(a.get_split_mfcc()) 
        a.get_split_frames()
        labels_list.append(a.get_split_labels()) 
        #a.get_plots()
    input_list = torch.cat(input_list)
    input_list = torch.transpose(input_list,1,2)
    labels_list = torch.from_numpy(np.concatenate(labels_list,axis = 0)).float()
    return input_list, labels_list

def load_data_limit(k=10000000,pkl_path = '/project/graziul/ra/ajays/whitelisted_vad_dict.pkl'): #load a subset of data
    #pkl_path = '/project/graziul/data/Zone1/2018_08_04/2018_08_04vad_dict.pkl'
    file = open(pkl_path,'rb')
    vad_dict = pickle.load(file)
    file.close()
    input_list = []
    labels_list = []

    for idx,key in enumerate(vad_dict):
        print(idx)
        if(idx == k):
            break
        a = audio_file(key)
        a.get_slices(vad_dict)
        input_list.append(a.get_split_mfcc()) 
        a.get_split_frames()
        labels_list.append(a.get_split_labels()) 
        #a.get_plots()
    input_list = torch.cat(input_list)
    input_list = torch.transpose(input_list,1,2)
    labels_list = torch.from_numpy(np.concatenate(labels_list,axis = 0)).float()
    return input_list, labels_list

def divide_audio(datafile, div_size = 30): #Divide the audio clip into bits of 1 minute each
#resizes input arrays from (1,feature_length, time) to (div_size,feature_length,time/div_length)
    return np.reshape(datafile,[div_size,datafile.shape[1],datafile.shape[2]//div_size])

class audio_file():
    def __init__(self, name,new_flag = 1):
        self.name = name
        self.vad_slices = None
        self.frames = None
        self.frames_labels = None
        self.mfcc = None
        self.n_clips = 300
        self.flag = new_flag
    
    def get_slices(self, vad_dict):
        if self.flag == 1:
            self.vad_slices = vad_dict[self.name]['nonsilent_slices']
        else:
            self.vad_slices = vad_dict[self.name]['pydub'][-24]['nonsilent_slices']
        return self.vad_slices
    
    def get_frames(self):
        ms_2_sample = self.sample_rate/1000
        frames_array = np.zeros(self.mfcc.shape[2])

        for v in self.vad_slices:
            start = math.floor(v[0]*ms_2_sample)
            end = math.ceil(v[1]*ms_2_sample)
            #print(v)
            for i in range(start,end):
                n = math.floor(i/220)
                j = i%220
                if j <= 110:
                    frames_array[n-2] += 1
                    frames_array[n-1] += 1
                    frames_array[n] += 1
                elif j>=111 and j<=220:
                    frames_array[n-1] += 1
                    frames_array[n] += 1
                elif j>=221 and j<=330:
                    frames_array[n-1] += 1
                    frames_array[n] += 1
                    frames_array[n+1] += 1
                elif j>=331 and j<=440:
                    frames_array[n+1] += 1
                    frames_array[n] += 1
                elif j>=441:
                    frames_array[n+2] += 1
                    frames_array[n+1] += 1
                    frames_array[n] += 1
            
        self.frames = frames_array
        return self.frames
        
    def get_split_frames(self):
        '''ms_2_sample = self.sample_rate/1000
        frame_arr_list = []
        for j in range(self.n_clips):
            frames_array = np.zeros(self.mfcc.shape[2])
            #frames_array = np.zeros(180409)
            self.clip_size = self.mfcc.shape[2]
            start_idx = j*self.clip_size
            end_idx = j*self.clip_size
            print(start_idx, end_idx)
            for v in self.vad_slices:
                start = math.floor(v[0]*ms_2_sample)
                end = math.ceil(v[1]*ms_2_sample)
                if(start >= start_idx and end <= end_idx):
                    for i in range(start,end):
                        n = math.floor(i/220)
                        j = i%220
                        if j <= 110:
                            frames_array[n-2] += 1
                            frames_array[n-1] += 1
                            frames_array[n] += 1
                        elif j>=111 and j<=220:
                            frames_array[n-1] += 1
                            frames_array[n] += 1
                        elif j>=221 and j<=330:
                            frames_array[n-1] += 1
                            frames_array[n] += 1
                            frames_array[n+1] += 1
                        elif j>=331 and j<=440:
                            frames_array[n+1] += 1
                            frames_array[n] += 1
                        elif j>=441:
                            frames_array[n+2] += 1
                            frames_array[n+1] += 1
                            frames_array[n] += 1
            frame_arr_list.append(np.expand_dims(frames_array,axis = 0))        
        self.frames = np.concatenate(frame_arr_list,axis = 0)
        return self.frames'''
        ms_2_sample = self.sample_rate/1000
        frames_array = np.zeros(self.mfcc.shape[2]*self.n_clips)
        print(frames_array.shape)

        for v in self.vad_slices:
            start = math.floor(v[0]*ms_2_sample)
            end = math.ceil(v[1]*ms_2_sample)
            for i in range(start,end):
                n = min(math.floor(i/220),len(frames_array)-1)
                j = i%220
                if j <= 110:
                    frames_array[n-2] += 1
                    frames_array[n-1] += 1
                    frames_array[n] += 1
                elif j>=111 and j<=220:
                    frames_array[n-1] += 1
                    frames_array[n] += 1
                elif j>=221 and j<=330:
                    frames_array[n-1] += 1
                    frames_array[n] += 1
                    frames_array[n+1] += 1
                elif j>=331 and j<=440:
                    frames_array[n+1] += 1
                    frames_array[n] += 1
                elif j>=441:
                    frames_array[n+2] += 1
                    frames_array[n+1] += 1
                    frames_array[n] += 1
        
        self.clip_size = self.mfcc.shape[2]
        frame_arr_list = []
        for j in range(self.n_clips):
            frame_arr_list.append(np.expand_dims(frames_array[j*self.clip_size:(j+1)*self.clip_size],axis=0))
        self.frames = np.concatenate(frame_arr_list,axis=0)
        return self.frames
    
        
    def get_labels(self): 
        self.frames_labels = np.zeros(len(self.frames))
        self.frames_labels[np.where(self.frames>0)] = 1
        return self.frames_labels
    
    def get_split_labels(self):
        self.frames_labels = np.zeros_like(self.frames)
        self.frames_labels[np.where(self.frames>0)] = 1
        return self.frames_labels
        
    def get_mfcc(self): 
        if self.flag == 0:
            file_name = '/project/graziul/data/Zone1/2018_08_04/' + self.name
        else:
            file_name = self.name
        self.waveform, self.sample_rate = torchaudio.load(file_name)
        pad_array = torch.zeros((1,10000*self.sample_rate))
        pad_array[:,:self.waveform.shape[1]] = self.waveform
        self.waveform = pad_array
        self.waveform = self.waveform[:,:1800*self.sample_rate] #Clip the file at 1800s
        n_fft = 2048
        win_length = 551
        hop_length = 220
        n_mels = 40
        n_mfcc = 40

        mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
              'n_fft': n_fft,
              'n_mels': n_mels,
              'hop_length': hop_length,
              'mel_scale': 'htk',
            }
        )

        self.mfcc = mfcc_transform(self.waveform)
        return self.mfcc
    
    def get_split_mfcc(self):
        if self.flag == 1:
            file_name = self.name
        else:
            file_name = '/project/graziul/data/Zone1/2018_08_04/' + self.name
        self.waveform, self.sample_rate = torchaudio.load(file_name)
        pad_array = torch.zeros((1,10000*self.sample_rate))
        pad_array[:,:self.waveform.shape[1]] = self.waveform
        self.waveform = pad_array
        self.waveform = self.waveform[:,:1800*self.sample_rate] #Clip the file at 1800s
        clip_size = math.floor(self.waveform.shape[1]/self.n_clips)
        n_clips = self.n_clips
        mfcc_list = []
        n_fft = 2048
        win_length = 551
        hop_length = 220
        n_mels = 40
        n_mfcc = 40
        mfcc_transform = T.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                  'n_fft': n_fft,
                  'n_mels': n_mels,
                  'hop_length': hop_length,
                  'mel_scale': 'htk',
                }
            )
        for i in range(n_clips):
            mfcc_list.append(mfcc_transform(self.waveform[:,i*clip_size:(i+1)*clip_size]))
        self.mfcc = torch.cat(mfcc_list)
        return self.mfcc
    
    def plot_waveform_with_labels(self,i,clip_size):
        plt.figure(figsize=(14,5))
        fig,(ax1,ax2) = plt.subplots(2,1)
        librosa.display.waveshow(self.waveform.squeeze().numpy()[i*clip_size:(i+1)*clip_size],self.sample_rate,ax = ax1)
        ax2.plot(self.frames_labels[i])
        plt.show()
        return    
    
    def get_plots(self): 
        clip_size = math.floor(1800*self.sample_rate/self.n_clips)
        for i in range(self.n_clips):
            print(i)
            self.plot_waveform_with_labels(i,clip_size)
        return
    
def plot_outputs(input_list, labels_list, output_hat, sample_rate = 22050):
    num_samples = input_list.size()[0]
    diff_labels = labels_list - output_hat
    for i in range(num_samples):
        print(i)
        plt.figure(figsize=(14,5))
        fig,(ax1,ax2) = plt.subplots(2,1)
        librosa.display.waveshow(input_list[i].numpy(),sample_rate,ax = ax1)
        ax2.plot(diff_labels[i].numpy())
        plt.show()
    return    
    
def get_predictions(model,input_list, labels_list, batch_size):
    output_list = []
    idx = 0
    num_samples = input_list.size()[0]//batch_size
    print(num_samples)
    with torch.no_grad():
        while(idx < num_samples):
            print(idx)
            input_batch = input_list[idx*batch_size:(idx+1)*batch_size]
            if labels_list is not None:
                labels_batch = labels_list[idx*batch_size:(idx+1)*batch_size]
            idx = idx+1
            output_hat = model(input_batch)
            #print(output_hat)
            #for param in model.parameters():
            #    print(param.grad)
            output_list.append(output_hat)
        output_list = torch.cat(output_list, dim = 0)
        return output_list

def get_frame_error_rate(output_hat, labels, verbose = 0):
    if verbose == 0:
        num_samples = labels.size()[0]
        fer_arr = []
        for i in range(num_samples):
            curr_output = output_hat[i]
            curr_label = labels[i]
            fer_arr.append(torch.mean(torch.add(curr_output,curr_label)%2).data*100)
        return fer_arr
    if verbose == 1:
        num_samples = labels.size()[0]
        fer_arr = []
        false_negatives = []
        false_positives = []
        for i in range(num_samples):
            false_positives.append(0)
            false_negatives.append(0)
            curr_output = output_hat[i]
            curr_label = labels[i]
            fer_arr.append(torch.mean(torch.add(curr_output,curr_label)%2).data*100)
            for j in range(len(curr_output)):
                my_curr_output = curr_output[j]
                my_curr_label = curr_label[j]
                if (my_curr_output == 1 and my_curr_label == 0):
                    false_positives[i] = false_positives[i] + 1
                if (my_curr_output == 0 and my_curr_label == 1):
                    false_negatives[i] = false_negatives[i] + 1
        return fer_arr,false_positives, false_negatives

def test_frame_error_rate(output_hat, labels, verbose = 0):
    if verbose == 0:
        num_samples = labels.size()[0]
        s_length = labels.size()[1]
        fer_arr = []
        sum = 0
        for i in range(num_samples):
            curr_output = output_hat[i]
            curr_label = labels[i]
            for j in range(s_length):
                if curr_output[j] == curr_label[j]:
                    pass
                else:
                    sum = sum+1
            fer_arr.append(torch.mean(torch.add(curr_output,curr_label)%2)*100)
        return sum
    else:
        num_samples = labels.size()[0]
        s_length = labels.size()[1]
        fer_arr = []
        false_positives = []
        false_negatives = []
        sum = 0
        for i in range(num_samples):
            false_positives[i] = 0
            false_negatives[i] = 0
            curr_output = output_hat[i]
            curr_label = labels[i]
            for j in range(s_length):
                if curr_output[j] == curr_label[j]:
                    pass
                else:
                    sum = sum+1
                if curr_output[j] == 1 and curr_label[j] == 0:
                    false_positives[i] = false_positives[i] + 1
                elif curr_output[j] == 0 and curr_label[j] == 1:
                    false_negatives[i] = false_negatives[i] + 1
            fer_arr.append(torch.mean(torch.add(curr_output,curr_label)%2)*100)
        return sum
