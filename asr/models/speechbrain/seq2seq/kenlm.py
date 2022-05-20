import kenl
import torch
import torch.nn as nn
from speechbrain.pretrained.interfaces import Pretrained

class KenLM(Pretrained):
    def __init__(self, modules=None, hparams=None, run_opts=None, freeze_params=True):
        self.super().__init__()

    def forward(self, x):
        self.mods.kenlm(x)

class KenLMModule(nn.Module):
    def __init__(self, pretrained_file):
        self.super().__init__()
        model = kenlm.LanguageModel(pretrained_file)
    
    def forward(self, x):
        with torch.no_grad():
            return model.score(x)

    
