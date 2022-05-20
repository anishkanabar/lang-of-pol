#!/usr/bin/env/python3
"""Recipe for training a DeepSpeech2-like system

Authors
 * Eric Chandler 2022
"""

import os
import sys
import torch
import logging
import librosa
import random
import pandas as pd
import speechbrain as sb
from speechbrain.nnet.activations import Softmax
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.WARNING)

class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        ids = batch.id
        tokens, tokens_lens = batch.tokens
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        
        # Spectrogram
        logger.debug(f'Wavs shape {wavs.shape}')
        x = self.modules.spec(wavs)

        # Masked ConvNet
        # XXX ILL NEVER FIX THESE PADDING ERRORS. JUST TRY TO REPLICATE DS WITH
        # THE SPEECHBRAIN CRDNN WHICH IS BASICALLY THE WHOLE THING-ISH
        logger.debug(f'Spec shape {x.shape}')
        h = self.modules.conv1(x)
        logger.debug(f'Conv 1 shape {h.shape}')
        h = self.modules.conv2(h)
        logger.debug(f'Conv 2 shape {h.shape}')
        

        # RNN

        # Lookahead + Linear
        # x = self.modules.lookahead(x)

        # Fully Connected
        # + Activation (here or in training loop?)
        logger.debug(f'Post-conv shape {h.shape}')
        probits = self.modules.fc(h)

        # Make Tokens Here or During Eval?
        if stage != sb.Stage.TRAIN:
            pred_tokens = sb.decoders.ctc_greedy_decode(probits, wav_lens, 
                blank_id=self.tokenizer.get_blank_index())
        else:
            pred_tokens = None
            
        return probits, wav_lens, pred_tokens

    def compute_objectives(self, predictions, targets, stage):
        probits, wav_lens, pred_tokens = predictions
        ids = targets.id
        tokens_eos, tokens_eos_lens = targets.tokens_eos
        tokens, tokens_lens = targets.tokens
    
        loss = self.hparams.ctc_cost(probits, tokens, wav_lens, tokens_lens)
        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                for utt_seq in pred_tokens
            ]
            target_words = [wrd.split(" ") for wrd in targets.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        
        return loss

    def fit_batch(self, batch):
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def init_optimizers(self):
        self.optimizer = self.hparams.optimizer(self.hparams.model.parameters())
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("model_opt", self.optimizer)

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()
        
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr_model)
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)
            with open(self.hparams.cer_file, "w") as w:
                self.cer_metric.write_stats(w)
            
           
        

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
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
    from ds_prepare import prepare_bpc, dataio_prepare  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_bpc,
        kwargs={
            "cluster": hparams["cluster"],
            "dataset_name": hparams['dataset_name'],
            "num_train": hparams["num_train"],
            "num_sec": hparams["num_sec"],
            "split_ratios": hparams["split_ratios"],
            "output_folder": hparams["output_folder"],
        },
    )
    
    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets, label_encoder = dataio_prepare(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = label_encoder
    
    # Training
    #with torch.autograd.detect_anomaly():
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        asr_brain.evaluate(
            test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
        )
