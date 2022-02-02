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

### Trial 11 - Deleted
Trained on Police - 1 hours
Uses Tokenizer BPE 128 Trial 10
Epochs: 40
FAILS even when write and corrupt check commented out.
      even when loading same number of audio files as LibriSpeech version
      the next thing I want to try is not subclassing utterance since most is commented out
     maybe it is actually because we load librosa library?
