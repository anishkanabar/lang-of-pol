import torchaudio
from torchaudio import transforms
import pickle
from speechbrain.pretrained import VAD
import os
import numpy as np

import pickle
pkl_path = '/project/graziul/ra/ajays/whitelisted_vad_dict.pkl'
file = open(pkl_path,'rb')
vad_dict = pickle.load(file)
file.close()

precision_list = []
recall_list = []
f1_list = []

class SpeechBrain(nn.module):
    def __init__(self):
        super(SpeechBrain,self)__init__()
        self.model = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="/project/graziul/ra/ajays/lang-of-pol/vad/Speechbrain/Models")
    
    def forward(self,x):
        path = '/project/graziul/ra/ajays/lang-of-pol/vad/Speechbrain/Data/test.wav'
        torchaudio.save(path, waveform, 16000, format='wav', encoding="PCM_S", bits_per_sample=16)
        boundaries = np.array(VAD.get_speech_segments(path))
        predictions = np.zeros(vad_dict[key]['nonsilent_slices'][-1][1])
        for n in range(boundaries.shape[0]):
            start = int(boundaries[n][0]*1000)
            end = int(boundaries[n][1]*1000)
            predictions[start:end] = 1  
        return predictions

for key in vad_dict.keys():
    waveform, sample_rate = torchaudio.load(key, normalize=True)
    transform = transforms.Resample(sample_rate, 16000)
    waveform = transform(waveform)
    title = os.path.basename(key)[:-4] 
    path = '/project/graziul/ra/anishk/VAD/Data/' + title + '.wav'
    print(path)
    torchaudio.save(path, waveform, 16000, format='wav', encoding="PCM_S", bits_per_sample=16)
#     VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
#     boundaries = np.array(VAD.get_speech_segments(path))

#     predictions = np.zeros(vad_dict[key]['nonsilent_slices'][-1][1])
#     for n in range(boundaries.shape[0]):
#         start = int(boundaries[n][0]*1000)
#         end = int(boundaries[n][1]*1000)
#         predictions[start:end] = 1  
#     #print(predictions)
    
#     truth = np.zeros(vad_dict[key]['nonsilent_slices'][-1][1])
#     for n in range(len(vad_dict[key]['nonsilent_slices'])):
#         start = vad_dict[key]['nonsilent_slices'][n][0]
#         end = vad_dict[key]['nonsilent_slices'][n][1]
#         truth[start:end] = 1                  
#     #print(truth)
    
#     precision = metrics.precision_score(truth, predictions)
#     precision_list.append(precision)
#     recall = metrics.recall_score(truth, predictions)
#     recall_list.append(recall)
#     f1 = metrics.f1_score(truth, predictions)
#     f1_list.append(f1)
