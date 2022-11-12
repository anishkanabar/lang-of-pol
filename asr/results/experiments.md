# Speechbrain

## Tokenizers
### BPE 128 - Trial 9
Trained on LibriSpeech - 1 hour

### BPE 128 - Trial 10
Trained on Police - 1 hour

### BPE 128 - Trial 11
Trained on LibriSpeech - 2 hour

### BPE 128 - Trial 12
Trained on police - 1:30 hours


## Seq2Seq
### Trial 9
Trained on LibriSpeech - 1 hour
Uses Tokenizer BPE 128 Trial 9
Epochs: 40
CER: 102

### Trial 10
Trained on LibriSpeech - 2 hours
Uses Tokenizer BPE 128 Trial 9
Epochs: 40
CER: 83.73

### Trial 11 - Deleted
Trained on Police - 1 hours
Uses Tokenizer BPE 128 Trial 10
Epochs: 40
FAILED even when write and corrupt check commented out.
      even when loading same number of audio files as LibriSpeech version
      even when commenting out librosa and soundfile import

### Trial 12
Trained on LibriSpeech - 4 hours
Uses Tokenizer BPE 128 Trial 11
FAILED Non-specific Cu-dnn runtime error in lstm.

### Trial 13
Trained on LibriSpeech - 4 hours - batch size 1
Uses Tokenizer BPE 128 Trial 11
Epochs: 40
CER: 199

### Trial 14
Trained on police - 1 hours - batch size 1
Uses Tokenizer BPE 128 Trial 11
Epochs: 40
CER: 97.53

### Trial 15
Trained on police - 4 hours - batch size 1
Uses Tokenizer BPE 128 Trial 11
FAILED maybe because expects >1 sample per audio, not >=1 ?

### Trial 16
Trained on police - 4 hours - batch size 1
Uses Tokenizer BPE 128 Trial 11
FAILED same as trial 15, after fixing filter to > 1

### Trial 17
Trained on police - 4 hours - batch size 1
Uses Tokenizer BPE 128 Trial 11
Overwrites utterances as 16000 Hz instead of 22050 Hz
(Cant figure out how to get 22050 to work with other
 speechbrain sizes like num fft)

### Trial 18, 19, 20
Trained on police - 2 hours - batch size 4, 2, 16
Uses Tokenizer BPE 128 Trial 11
FAILED for random cuda bs

### Trial 21
Trying to recreate trial 14
FAILED for random cuda bs

### Trial 22
Trained on librispeech 4 hours
Uses Tokenizer BPE 128 Trial 11
Batch Size: 8
CER: 273

### Trial 23
Trained on librispeech 8 hours
Uses Tokenizer BPE 128 Trial 11
Batch Size: 8

### Trial 24
Trained on atczero 8 hours
Uses Tokenizer BPE 128 Trial 11
Batch Size: 8

### Trial 25
Trained on police 8 hours
Uses Tokenizer BPE 128 Trial 11
Batch Size: 8

## CTC
### Trial 3
Dataset: police
Duration: 2 hours
FAILED for encoder broken graph

### Trial 4
Dataset: librispeech
Duration: 2 hours
FAILED for encoder broken graph

