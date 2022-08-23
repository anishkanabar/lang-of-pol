import datasets
import transformers
from datasets import load_dataset, Audio, load_metric
from transformers import WavLMForCTC, TrainingArguments, Trainer, HfArgumentParser, set_seed
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


# Things to add: Logging details 

def extract_all_chars(batch):
	all_text = " ".join(batch["wrd"])
	vocab = list(set(all_text))
	
	return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset(batch):
	audio = batch["wav"]
	batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
	batch["input_length"] = len(batch["input_values"])
	#with processor.as_target_processor():
	#		batch["labels"] = processor(batch["wrd"]).input_ids
		
	batch["labels"] = tokenizer(batch["wrd"]).input_ids
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
	# Step 1: Loading Files 
	data_files = {"train": hparams["train_csv"], "evaluation": hparams["valid_csv"]}
	dataset = load_dataset("csv", data_files=data_files) 
	
	
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
	feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
	
	print("Starting Processor")
	processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
	
	print("Casting Audio column")
	dataset_samp = dataset.cast_column("wav", Audio(sampling_rate=16000))
	
	print("Preparing dataset")
	dataset_samp = dataset_samp.map(prepare_dataset, remove_columns=dataset_samp.column_names["train"], num_proc=4)
	
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
		
		final_string = "[wer: {}, pred_str: {}, actual_str:{}]".format(wer, pred_str, label_str)
		
		with open(hparams["wer_file"], "w") as w:
			w.write(final_string)
				
		return {"wer": wer}
	
	
	model_args = hparams["model_args"]
	data_args = hparams["data_train_args"]
	train_args = hparams["training_args"]
	parser = HfArgumentParser(TrainingArguments)
	train_args = parser.parse_dict({**train_args})
	
	print("Setting up Model")
	model = WavLMForCTC.from_pretrained(
		hparams["wavlm_hub"],
		ctc_loss_reduction="mean",
		pad_token_id=processor.tokenizer.pad_token_id,
		vocab_size=len(processor.tokenizer) ## Using this because the default value is set to 32. but our text has numbers as well, increasing the size of our vocabulary 
		)
		
	if model_args["freeze_feature_encoder"]:	
		model.freeze_feature_encoder()
	
	max_input_length = data_args["max_duration_in_seconds"] * feature_extractor.sampling_rate
	min_input_length = data_args["min_duration_in_seconds"] * feature_extractor.sampling_rate
	audio_column_name = data_args["audio_column_name"]
	num_workers = data_args["preprocessing_num_workers"]
	

	def is_audio_in_length_range(length):
	    return length > min_input_length and length < max_input_length

	# filter data that is shorter than min_input_length
	vectorized_datasets = dataset_samp.filter(
	    is_audio_in_length_range,
	    num_proc=num_workers,
	    input_columns=["input_length"],
	)
	
	
	# save feature extractor, tokenizer and config
	feature_extractor.save_pretrained(train_args[0].output_dir)
	tokenizer.save_pretrained(train_args[0].output_dir)
	#config.save_pretrained(training_args.output_dir)
	
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
		#if last_checkpoint is not None:
		#    checkpoint = last_checkpoint
		#elif os.path.isdir(model_args.model_name_or_path):
		#    checkpoint = model_args.model_name_or_path
		#else:
		#    checkpoint = None
		#print(checkpoint)
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
		#logger.info("*** Evaluate ***")
		metrics = trainer.evaluate()
		max_eval_samples = len(vectorized_datasets["evaluation"])
		metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))

		trainer.log_metrics("eval", metrics)
		trainer.save_metrics("eval", metrics)
	
	print(results)
	
