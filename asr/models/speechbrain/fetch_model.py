#from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2

#model_name = "facebook/wav2vec2-base"
#model_name = "speechbrain/asr-wav2vec2-commonvoice-en"
#model_name = "speechbrain/asr-crdnn-rnnlm-librispeech"
#model_name = "facebook/wav2vec2-large-960h-lv60-self"
#model_name = "microsoft/wavlm-large"
model_name = "patrickvonplaten/wavlm-libri-clean-100h-large"
revision = "main"

#save_path = "/project/graziul/ra/echandler/scratch/" + model_name
save_path = "/project/graziul/ra/pshroff/scratch/" + model_name
#save_path = "/scratch/midway3/echandler/" + model_name
output_norm = True
freeze = True
#HuggingFaceWav2Vec2(model_name, save_path, output_norm, freeze)

from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download(repo_id=model_name, revision=revision, cache_dir=save_path)
