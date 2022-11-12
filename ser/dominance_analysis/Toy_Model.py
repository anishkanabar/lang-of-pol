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

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.input_dim1 = 40
        self.input_dim2 = 64 
        self.hidden_dim = 64
        self.n_layers = 3
        self.batch_size = 2
        #(input is of format batch_size, sequence_length, num_features)
        #hidden states should be (num_layers, batch_size, hidden_length)
        self.hidden_state1 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        self.cell_state1 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        self.hidden_state2 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        self.cell_state2 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        self.lstm1 = nn.LSTM(input_size = self.input_dim1, hidden_size = self.hidden_dim, num_layers = self.n_layers, batch_first=True) #should be True
        self.lstm2 = nn.LSTM(input_size = self.input_dim2, hidden_size = self.hidden_dim, num_layers = self.n_layers, batch_first=True) #should be True
        self.lstm2_out = None 
        self.hidden = None
        #self.flatten = nn.Flatten()
        self.convolve1d = nn.Sequential(
            nn.Conv1d(3,3, kernel_size=11, padding=5),
            nn.BatchNorm1d(64, affine=False, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(3,5, kernel_size=11, padding=5),
            nn.BatchNorm1d(64, affine=False, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(5,5, kernel_size=11, padding=5),
            nn.BatchNorm1d(64, affine=False, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(5,1, kernel_size=11, padding=5)
        )
        self.output_stack = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 1)
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, data):
        out1,_ = self.lstm1(data,(self.hidden_state1,self.cell_state1))
        out1,_ = self.lstm2(out1,(self.hidden_state2,self.cell_state2))
        out2 = self.sigmoid(self.output_stack(out1))
        return torch.squeeze(out2)
        
