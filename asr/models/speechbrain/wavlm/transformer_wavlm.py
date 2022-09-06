import datasets
import transformers
from datasets import load_dataset, Audio, load_metric
from transformers import WavLMForCTC, TrainingArguments, Trainer, HfArgumentParser, set_seed, WavLMConfig
import torch 
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from dataclasses import dataclass, field 
from typing import Any, Dict, List, Optional, Union 
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
import os
import sys
import torch
import logging
import librosa
import pandas as pd
import numpy as np
import json 
import soundfile as sf
import re 
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

 
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.22.0.dev0")

require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")


logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)
    
    
def extract_all_chars(batch):
	all_text = " ".join(batch["wrd"])
	vocab = list(set(all_text))
	
	return {"vocab": [vocab], "all_text": [all_text]}

def output_predictions(pred, labels, wer):

	output_string = """
	
	--------------------------------------
	Word Error Rate: {}
	--------------------------------------
	
	|     Prediction      |      Actual Label      | 
	
	""".format(round(wer*100, 3))
	
	for i in range(len(pred)):
	
		val = """
		--------------------------------------
		
		| {} | {} |
		
		--------------------------------------
		
		""".format(pred[i], labels[i])
		
		output_string = output_string + val 
		
	
	return output_string
	
def check_corrupt(wav):

	try:
		sf.read(wav)
		return False 
	except:
		return True
	
def filter_corrupt(df):
	
	unique_paths = pd.Series(df["wav"].unique())
	corrupt_map = {}
	for unique_path in unique_paths:
		corrupt_map[unique_path] = not check_corrupt(unique_path)
	
	mp3_notcorrupt = df["wav"].transform(lambda p: corrupt_map[p])
	n_corrupted = mp3_notcorrupt.count() - mp3_notcorrupt.sum()
	print("Discarding {} corrupted mp3s".format(n_corrupted))
	
	return df.loc[mp3_notcorrupt]

def hasNumber(inputString):
	return bool(re.search(r'\d', inputString))
	
def filter_numbers(df):
	
	df["numbers"] = df["wrd"].apply(lambda x: hasNumber(x))
	zero = df[df["numbers"]==False].shape[0]
	total = df.shape[0]
	
	print("Discarding {} files with numbers in transcriptions".format(total-zero))
	
	return df.loc[df["numbers"]==False, :]
	
def filter_df(df, val):

	df = filter_numbers(df)
	df = filter_corrupt(df)
	
	df.to_csv(hparams["output_folder"]+"/"+val)
					
		
def prepare_dataset(batch):
	
	audio = batch["wav"]
	batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
	batch["input_length"] = len(batch["input_values"])
	with processor.as_target_processor():
		batch["labels"] = processor(batch["wrd"]).input_ids

	#batch["labels"] = tokenizer(batch["wrd"]).input_ids
	return batch 

		
@dataclass 
class DataCollatorCTCWithPadding:
	processor: Wav2Vec2Processor

	padding: Union[bool, str] = True
	max_length: Optional[int] = None
	max_length_labels: Optional[int] = None
	pad_to_multiple_of: Optional[int] = None
	pad_to_multiple_of_labels: Optional[int] = None
	
	def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
		input_features = [{"input_values": feature["input_values"]} for feature in features]
		label_features = [{"input_ids": feature["labels"]} for feature in features]

		batch = self.processor.pad(
		    input_features,
		    padding=self.padding,
		    max_length=self.max_length,
		    pad_to_multiple_of=self.pad_to_multiple_of,
		    return_tensors="pt",
		)
		with self.processor.as_target_processor():
		    labels_batch = self.processor.pad(
		        label_features,
		        padding=self.padding,
		        max_length=self.max_length_labels,
		        pad_to_multiple_of=self.pad_to_multiple_of_labels,
		        return_tensors="pt",
		    )

		# replace padding with -100 to ignore loss correctly
		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

		batch["labels"] = labels

		return batch		
		
	
if __name__ == "__main__":
	
	
	# CLI:
	hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

	# If distributed_launch=True then
	# create ddp_group with the right communication protocol
	sb.utils.distributed.ddp_init_group(run_opts)

	with open(hparams_file) as fin:
		hparams = load_hyperpyyaml(fin, overrides)

	# Create experiment directory
	sb.create_experiment_directory(
		experiment_directory=hparams["output_folder"],
		hyperparams_to_save=hparams_file,
		overrides=overrides,
	)

	# Dataset prep (parsing Librispeech)
	from ctc_prepare import create_manifests, dataio_prepare  # noqa

	# multi-gpu (ddp) save data preparation
	run_on_main(
		create_manifests,
		kwargs={
		    "cluster": hparams["cluster"],
		    "dataset_name": hparams['dataset_name'],
		    "splits": hparams["splits"],
		    "output_folder": hparams["output_folder"],
		    "ambiguity_strategy": hparams['ambiguity_strategy'],
		    "skip_prep": hparams["skip_prep"],
		    "seed": hparams["seed"]
		    #"stratify": hparams["stratify"],
		},
	)
	
	set_seed(hparams["seed"])
	
	# Removing corrupt files 
	filter_df(pd.read_csv(hparams["train_csv"]), "training.csv")
	filter_df(pd.read_csv(hparams["valid_csv"]), "validation.csv")
	
	# Reading in all the arguments 
	model_args = hparams["model_args"]
	data_args = hparams["data_train_args"]
	train_args = hparams["training_args"]
	parser = HfArgumentParser(TrainingArguments)
	train_args = parser.parse_dict({**train_args})
	audio_column_name = data_args["audio_column_name"]
	num_workers = data_args["preprocessing_num_workers"]
	
	
	# Detecting last checkpoint.
	last_checkpoint = None
	if os.path.isdir(train_args[0].output_dir) and train_args[0].do_train and not train_args[0].overwrite_output_dir:
		last_checkpoint = get_last_checkpoint(train_args[0].output_dir)
		if last_checkpoint is None and len(os.listdir(train_args[0].output_dir)) > 0:
		    raise ValueError(
		        f"Output directory ({training_args.output_dir}) already exists and is not empty. "
		        "Use --overwrite_output_dir to overcome."
		    )
		elif last_checkpoint is not None:
		    logger.info(
		        f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
		        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
		    )
	print("Last checkpoint: {}".format(last_checkpoint))
	
	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		handlers=[logging.StreamHandler(sys.stdout)],
	)
	logger.setLevel(logging.INFO if is_main_process(train_args[0].local_rank) else logging.WARN)
	
	# Set the verbosity to info of the Transformers logger (on main process only):
	if is_main_process(train_args[0].local_rank):
		transformers.utils.logging.set_verbosity_info()
	logger.info("Training/evaluation parameters %s", train_args[0])


	# Step 1: Loading Files  
	
	#data_files = {"train": hparams["train_csv"], "evaluation": hparams["valid_csv"]}
	data_files = {"train": hparams["output_folder"]+"/training.csv", "evaluation": hparams["output_folder"]+"/validation.csv"}
	dataset = load_dataset("csv", data_files=data_files) 
	
	# sorting data 
	"""
	try:
		if hparams["sorting"] == "ascending":
			reverse=False
		else:
			reverse=True 
	except:
		reverse=False
		
	dataset.sort("duration", reverse=reverse)
	"""
	
	# Step 2: Creating vocab.json
	vocabs = dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset.column_names["train"])
	
	vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["evaluation"]["vocab"][0]))
	vocab_dict = {v: k for k,v in enumerate(vocab_list)}
	
	vocab_dict["|"] = vocab_dict[" "]
	del vocab_dict[" "]
	
	vocab_dict["[UNK]"] = len(vocab_dict)
	vocab_dict["[PAD]"] = len(vocab_dict)
	
	vocab_file = hparams["output_folder"] + "/vocab.json"
	with open(vocab_file, 'w') as vocab_files:
		json.dump(vocab_dict, vocab_files) 
	
	
	print("Starting Tokenizer")	
	tokenizer = Wav2Vec2CTCTokenizer(vocab_file, unk_token="[UNK]", pad_token="[PAD]", word_delimeter_token="|")
	
	print("Starting feature extractor")
	feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
	
	print("Starting Processor")
	processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
	
	print("Casting Audio column")
	dataset_samp = dataset.cast_column("wav", Audio(sampling_rate=16000))
	
	print(dataset_samp["train"][0])
	print(type(dataset))
	
	print("Preparing dataset")
	
	vectorized_datasets = dataset_samp.map(prepare_dataset, remove_columns=dataset_samp.column_names["train"], num_proc=num_workers)
	
	print("Data Collator")
	data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
	
	wer_metric = load_metric("wavlm/metrics/wer", cache_dir=hparams["output_folder"])
	
	
	def compute_metrics(pred):
		pred_logits = pred.predictions
		pred_ids = np.argmax(pred_logits, axis=-1)
		
		pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id 
		
		pred_str = processor.batch_decode(pred_ids)
		
		label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
		wer = wer_metric.compute(predictions=pred_str, references=label_str)
		
		#final_string = "[wer: {}, pred_str: {}, actual_str:{}]".format(wer, pred_str, label_str)
		final_string = output_predictions(pred_str, label_str, wer)
		
		with open(hparams["wer_file"], "w") as w:
			w.write(final_string)
		
		
		return {"wer": wer}
	
	
	
	
	print("Setting up Config")
	config = WavLMConfig.from_pretrained(hparams["wavlm_hub"], cache_dir=hparams["output_folder"])
	
	config.update(
        {
            "hidden_dropout": model_args["hidden_dropout"],
            "attention_dropout": model_args["attention_dropout"],
            "ctc_loss_reduction": "mean",
            "pad_token_id": processor.tokenizer.pad_token_id,
            "vocab_size": len(processor.tokenizer) ## Using this because the default value is set to 32. but our text has numbers as well, increasing the size of our vocabulary 
        }
    )
	
	print("Setting up Model")
	model = WavLMForCTC.from_pretrained(
		hparams["wavlm_hub"],
		config=config
		)
		
	if model_args["freeze_feature_encoder"]:	
		model.freeze_feature_encoder()
	
	print("Output Embeddings")
	print(model.get_output_embeddings)
	
	
	#max_input_length = data_args["max_duration_in_seconds"] * feature_extractor.sampling_rate
	#min_input_length = data_args["min_duration_in_seconds"] * feature_extractor.sampling_rate

	# save feature extractor, tokenizer and config
	feature_extractor.save_pretrained(train_args[0].output_dir)
	tokenizer.save_pretrained(train_args[0].output_dir)
	config.save_pretrained(train_args[0].output_dir)
	
	trainer = Trainer(
		model=model,
		args=train_args[0],
		train_dataset=vectorized_datasets["train"],
		eval_dataset=vectorized_datasets["evaluation"],
		tokenizer=processor.feature_extractor,
		compute_metrics=compute_metrics,
		data_collator=data_collator
		)
		
	print("Starting Training")
	
	if train_args[0].do_train:

		# use last checkpoint if exist
		if last_checkpoint is not None:
		    checkpoint = last_checkpoint
		#elif os.path.isdir(model_args.model_name_or_path):
		#    checkpoint = model_args.model_name_or_path
		else:
		    checkpoint = None
		print(checkpoint)
		train_result = trainer.train()
		trainer.save_model()

		metrics = train_result.metrics
		max_train_samples = len(vectorized_datasets["train"])
		metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))

		trainer.log_metrics("train", metrics)
		trainer.save_metrics("train", metrics)
		trainer.save_state()

	# Evaluation
	results = {}
	if train_args[0].do_eval:
		logger.info("*** Evaluate ***")
		metrics = trainer.evaluate()
		max_eval_samples = len(vectorized_datasets["evaluation"])
		metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["evaluation"]))

		trainer.log_metrics("eval", metrics)
		trainer.save_metrics("eval", metrics)
	
	print(results)
	
	
