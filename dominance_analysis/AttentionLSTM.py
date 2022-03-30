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

class Attention_LSTM(nn.Module):
    def __init__(self):
        super(Attention_LSTM, self).__init__()
        self.input_dim1 = 40
        self.input_dim2 = 64 
        self.hidden_dim = 64
        self.n_layers = 3
        self.batch_size = 2
        self.unroll=0
        self.unroll_steps = 50
        #(input is of format batch_size, sequence_length, num_features)
        #hidden states should be (num_layers, batch_size, hidden_length)
        self.hidden_state1 = torch.randn(1, self.batch_size, self.hidden_dim)
        self.cell_state1 = torch.randn(1, self.batch_size, self.hidden_dim)
        self.lstm1 = nn.LSTM(input_size = self.input_dim1, hidden_size = self.hidden_dim, num_layers = 1, batch_first=True) #should be True
        self.lstm2 = nn.LSTM(input_size = self.input_dim2, hidden_size = self.hidden_dim, num_layers = 1, batch_first=True) #should be True
        #self.flatten = nn.Flatten()
        self.convolve1d = nn.Sequential(
            nn.Conv1d(3,3, kernel_size=11, padding=5),
            nn.BatchNorm1d(5, affine=False, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(3,5, kernel_size=11, padding=5),
            nn.BatchNorm1d(5, affine=False, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(5,5, kernel_size=11, padding=5),
            nn.BatchNorm1d(5, affine=False, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(5,1, kernel_size=11, padding=5),
            nn.BatchNorm1d(1, affine=False, track_running_stats=False)
        )
        self.output_stack = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 1)
        )
        self.sigmoid = nn.Sigmoid()

#     def create_rand_hidden1(self):
#         self.hidden_state1 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
#         self.cell_state1 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
#         return (self.hidden_state1, self.cell_state1)

    def lstm1_func(self,data, idx):
        if(self.unroll ==0):
            if(idx == 0):
                H, (self.hidden_state1, self.cell_state1) = self.lstm1(data) 
                hidden = (self.hidden_state1, self.cell_state1)
            else:
                H, (self.hidden_state1, self.cell_state1) = self.lstm2(data) 
                hidden = (self.hidden_state1, self.cell_state1)
        else:
            hn = self.hidden_state1
            cn = self.cell_state1
            H = []
            for idx in range(data.size()[1]//self.unroll_steps + 1):
                output, (hn,cn) = self.lstm1(data[:,idx*self.unroll_steps:(idx+1)*self.unroll_steps,:], (hn,cn))
                H.append(output)
            H = torch.cat(H,dim=1)
            hidden = (hn,cn)
            print(H.size())
        return H,hidden

    def temp_attention(self, data,idx):
        H,hidden = self.lstm1_func(data,idx)
        H_maxtemp = torch.unsqueeze(torch.max(H, -1).values,2)
        H_avgtemp = torch.unsqueeze(torch.mean(H, -1),2)
        H_stdtemp = torch.unsqueeze(torch.std(H, -1),2)
        H_concattemp = torch.cat([H_maxtemp, H_avgtemp,H_stdtemp], dim=2)
        H_concattemp = torch.transpose(H_concattemp, 1,2)
        #print(H_concattemp.size())
        return H_concattemp,H 
    
    def convolve1(self, data,idx):
        H_concattemp,H = self.temp_attention(data,idx)
        H_temp = self.convolve1d(H_concattemp)
        # "Expand/copy" output of last layer (H_temp) to same dims as H
        H_temp = H_temp.expand(-1,64,-1)
        H_temp = torch.transpose(H_temp,1,2)
        # Sigmoid activation     
        sigmoid = nn.Sigmoid()
        my_input = H_temp
        H_temp = sigmoid(my_input)
        # Merge H_temp and H by element wise summation
        H_prime = H + H_temp
        return H_prime
        
#     def create_rand_hidden2(self):
#         self.hidden_state2 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
#         self.cell_state2 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
#         return (self.hidden_state2, self.cell_state2)  
    
#     def freq_attention(hidden_feature_map):
#         H_maxfreq = torch.max(hidden_feature_map, 0).values
#         H_avgfreq = torch.mean(hidden_feature_map, 0)
#         H_stdfreq = torch.std(hidden_feature_map, 0)
#         H_concatfreq = torch.cat([H_maxfreq[None, :], H_avgfreq[None, :], H_stdfreq[None,:]], dim=0)
#         return H_concatfreq 


    def forward(self, data):
        output = data
        for i in range(self.n_layers):
            output = self.convolve1(output,i)
        self.output = self.output_stack(output)
        #print(self.output)
        self.output = torch.squeeze(self.sigmoid(self.output))
        return self.output
        
