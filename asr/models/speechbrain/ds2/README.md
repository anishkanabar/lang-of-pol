# DeepSpeech 2 implementation using SpeechBrain framework

## Goal
Re-implement DeepSpeech 2 architecture using SpeechBrain components

## Reasoning
I wanted another architecture for benchmarkf performance. 
We finally have a stable codebase in the SpeechBrain ecosystem such that
it shouldn't be a major lift to do this.

## Status
Abandoned because the adjacent 'seq2seq' model is VERY similar:
spec augment + rnn + dnn + ctc loss
While it's not exactly the same, I thought it best to prioritize the pre-defined seq2seq model.
