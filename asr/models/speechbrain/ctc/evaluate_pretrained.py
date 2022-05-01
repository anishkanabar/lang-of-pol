from speechbrain.pretrained import EncoderDecoderASR

pretrained_file = '/project/graziul/ra/echandler/scratch/speechbrain/asr-wav2vec2-commonvoice-en/speechbrain--asr-wav2vec2-commonvoice-en.main.d517e2f6b4983559a6365988534105b41edb8ae9'

asr_model = EncoderDecoderASR.from_hparams(source=pretrained_file)

audio_file = '/project/graziul/data/utterances/Zone8/2018_08_10/201808101202-916683-27158/771512_772491.flac'
text = asr_model.transcribe_file(audio_file)
print(text)
