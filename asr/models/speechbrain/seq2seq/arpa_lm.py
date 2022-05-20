import torch
import torch.nn as nn
from speechbrain.lm.ngram import BackoffNgramLM, read_arpa

class KenLM(nn.Module):
    def __init__(self, pretrained_file):
        self.super().__init__()
        num_grams, ngrams, backoffs = read_arpa(pretrained_file)
        self.lm = BackoffNgramLM(ngrams, backoffs)
    
    def forward(self, x):
        with torch.no_grad():
            return self.lm.logprob(x)

