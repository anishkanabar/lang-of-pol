#from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2

model_name = "facebook/wav2vec2-base"
revision = "main"

save_path = "/project/graziul/ra/echandler/scratch/" + model_name
#save_path = "/scratch/midway3/echandler/" + model_name
output_norm = True
freeze = True
#HuggingFaceWav2Vec2(model_name, save_path, output_norm, freeze)

from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download(model_name, revision, save_path)
