import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchaudio
import sys
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
import util
from util import audio_file
from util import *
from Losses import FocalLoss,WeightedFocalLoss
from StackedLSTM2 import StackedLSTM
from AttentionLSTM import Attention_LSTM
from Toy_Model import ToyModel
import time
torch.manual_seed(1)

my_dataset = sys.argv[1]
my_model = sys.argv[2]
verbose = int(sys.argv[3])

n_samples = 40
train_split = 4*n_samples//5
test_samples = n_samples - train_split
if my_dataset == "ATC0":
    input_list, labels_list = process_atc0_files()
    
elif my_dataset == "BPC":
    input_list, labels_list = load_data()
    
test_input_list = input_list[30*train_split:]
test_labels_list = labels_list[30*train_split:]
input_list = input_list[:train_split*30]
labels_list = labels_list[:train_split*30]

if my_model == "Attention_LSTM":
    model = Attention_LSTM()
    save_filepath = '/project/graziul/ra/ajays/LSTM_model_predictions.txt'
elif my_model == "Vanilla_LSTM":
    model = Toy_Model()
    save_filepath = '/project/graziul/ra/ajays/toy_model_predictions.txt'
loss_fn = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
fer_list = []
train_loss_list = []
test_loss_list = []
batch_size = model.batch_size
num_samples = input_list.size()[0]//batch_size
training_steps = 1000
idx = 0
flag = 0
num_segments = 30
val_size = 30
'''for step in range(training_steps):
    start_time = time.time()
    input_batch = input_list[idx*batch_size:(idx+1)*batch_size]
    labels_batch = labels_list[idx*batch_size:(idx+1)*batch_size]
    idx = (idx+1)%num_samples
    print(step)
    optimizer.zero_grad()
    output_hat = model(input_batch)
    #print(output_hat)
    print(labels_batch)
    print(output_hat)
    loss = loss_fn(output_hat, labels_batch)
    loss.backward()
    #for param in model.parameters():
    #    print(param.grad)
    print(loss)
    train_loss_list.append(loss.item())
    optimizer.step()
    end_time = time.time()
    step_time = end_time - start_time
    print("Time Taken for Step = " + str(step_time))
    if step%num_segments == 0:
        with torch.no_grad():
            preds = get_predictions(model,test_input_list, test_labels_list, batch_size)
            err = get_frame_error_rate(torch.round(preds),test_labels_list)
            test_loss = loss_fn(preds, test_labels_list)
            test_loss_list.append(test_loss)
            fer = str(err) + "\n"
            print(fer)
            fer_list.append(torch.mean(torch.stack(err)))

plt.plot(list(range(training_steps//num_segments)),fer_list)
plt.savefig('LSTM_model_training.png')
'''
preds_file = open(save_filepath,'w')
preds_file.truncate(0)
for val_index in range(num_samples):
    #model = StackedLSTM()
    #model = Attention_LSTM()
    if my_model == "Attention_LSTM":
        model = Attention_LSTM()
    elif my_model == "Vanilla_LSTM":
        model = ToyModel()
    #model = nn.DataParallel(model)
    loss_fn = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    test_data = input_list[val_index*val_size:(val_index+1)*val_size] 
    test_labels = labels_list[val_index*val_size:(val_index+1)*val_size]
    train_data = torch.cat([input_list[:val_index*val_size], input_list[(val_index+1)*val_size:]], dim = 0)
    train_labels = torch.cat([labels_list[:val_index*val_size], labels_list[(val_index+1)*val_size:]], dim = 0)
    print(val_index)
    idx = 0
    for step in range(training_steps):
        if(idx != val_index):
            input_batch = train_data[idx*batch_size:(idx+1)*batch_size]
            labels_batch = train_labels[idx*batch_size:(idx+1)*batch_size]
            print(step)
            optimizer.zero_grad()
            output_hat = model(input_batch)
            print(labels_batch)
            print(output_hat)
            loss = loss_fn(output_hat, labels_batch)
            loss.backward()
            #for param in model.parameters():
            #    print(param.grad)
            print(loss)
            optimizer.step()
        idx = (idx+1)%num_samples
    with torch.no_grad():
        preds = get_predictions(model,test_data, test_labels, batch_size)  
        fer_value = get_frame_error_rate(torch.round(preds),test_labels, verbose)
        if verbose == 0:
            fer = str(fer_value) + "\n"
        else:
            fer = "FER = " + str(fer_value[0]) + ", False_Positives = " + str(fer_value[1]) + ", False Negatives = " + str(fer_value[3]) + "\n"
        print(fer)
        if verbose == 0:
            preds_file.write(fer)
preds_file.close()

