# Speechbrain

## Tokenizers
### BPE 128 - Trial 9
Trained on LibriSpeech - 1 hour

### BPE 128 - Trial 10
Trained on Police - 1 hour

### BPE 128 - Trial 11
Trained on LibriSpeech - 2 hour

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

### Trial 14
Trained on police - 1 hours - batch size 1
Uses Tokenizer BPE 128 Trial 11

### Trial 15
Trained on police - 4 hours - batch size 1
Uses Tokenizer BPE 128 Trial 11
FAILED maybe because expects >1 sample per audio, not >=1 ?

### Trial 16
Trained on police - 4 hours - batch size 1
Uses Tokenizer BPE 128 Trial 11
