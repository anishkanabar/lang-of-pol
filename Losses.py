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

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=1):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

    
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=1,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, inputs, targets):
        #print(inputs)
        #print(targets)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return torch.mean(BCE_loss)
        #ce_loss = F.binary_cross_entropy(inputs, targets,reduction=self.reduction,weight=self.weight)
        #pt = torch.exp(-ce_loss)
        #focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        #return focal_loss