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

class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        ids = batch.id
        tokens, tokens_lens = batch.tokens
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        real_lens = (tokens != 0).sum(dim=1)
        
        # Spectrogram
        logger.debug('forward spec')
        x = self.modules.spec(wavs)
        assert torch.isfinite(x).all(), "spectrum not finite"
        # logger.debug("CHECK,ID,WAV_LEN,MFCC_LEN,CHAR_LEN")
        # for i in range(len(ids)):
        #     logger.debug(f"CHECK,ID,{ids[i]},WAV_LEN,{wavs[i].shape[0]},MFCC_LEN,{x[i].shape},CHAR_LEN{tokens[i].shape}")
        # Masked ConvNet

        # RNN

        # Lookahead + Linear
        # x = self.modules.lookahead(x)

        # Fully Connected
        # + Activation (here or in training loop?)
        pred_tokens = None
        logger.debug('forward fc')
        # for p in self.modules.fc.parameters():
        #     assert torch.isfinite(p).all(), "fc not finite"
        probits = self.modules.fc(x)
        # assert torch.isfinite(probits).all(), "probits not finite"

        # Make Tokens Here or During Eval?
        # XXX: Is it problematic to use static blank index if that stuff gets moved?
        if stage != sb.Stage.TRAIN:
            pred_tokens = sb.decoders.ctc_greedy_decode(probits, wav_lens, 
                blank_id=self.tokenizer.get_blank_index())
            
        return probits, wav_lens, pred_tokens

    # def viterbi_decode(self, feats):
    #     backpointers = []

    #     # Initialize the viterbi variables in log space
    #     print(f'feats shape {feats.shape}')
    #     init_vvars = torch.full((1, hparams['num_classes']), -10000.).to(self.device)
    #     init_vvars[0][self.tokenizer.get_bos_index()] = 0

    #     # forward_var at step i holds the viterbi variables for step i-1
    #     forward_var = init_vvars
    #     for feat in feats:
    #         bptrs_t = []  # holds the backpointers for this step
    #         viterbivars_t = []  # holds the viterbi variables for this step

    #         for next_tag in range(hparams['num_classes']):
    #             # next_tag_var[i] holds the viterbi variable for tag i at the
    #             # previous step, plus the score of transitioning
    #             # from tag i to next_tag.
    #             # We don't include the emission scores here because the max
    #             # does not depend on them (we add them in below)
    #             next_tag_var = forward_var + feats[next_tag]
    #             best_tag_id = torch.argmax(next_tag_var)
    #             bptrs_t.append(best_tag_id)
    #             viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
    #         # Now add in the emission scores, and assign forward_var to the set
    #         # of viterbi variables we just computed
    #         forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
    #         backpointers.append(bptrs_t)

    #     # Transition to STOP_TAG
    #     terminal_var = forward_var + feats[self.tokenizer.get_eos_index()]
    #     best_tag_id = torch.argmax(terminal_var)

    #     # Follow the back pointers to decode the best path.
    #     best_path = [best_tag_id]
    #     for bptrs_t in reversed(backpointers):
    #         best_tag_id = bptrs_t[best_tag_id]
    #         best_path.append(best_tag_id)
    #     # Pop off the start tag (we dont want to return that to the caller)
    #     best_path.reverse()
    #     return sb.decoders.ctc.filter_ctc_output(best_path)

    def compute_objectives(self, predictions, targets, stage):
        probits, wav_lens, pred_tokens = predictions
        ids = targets.id
        tokens_eos, tokens_eos_lens = targets.tokens_eos
        tokens, tokens_lens = targets.tokens
    
        # HACK VITERBI
        # pred_seqs = self.viterbi_decode(probits)
        # wer, _, _, _, _ = sb.utils.edit_distance.accumulatable_wer_stats(
        #     tokens, pred_seqs
        # )
        # loss = wer
        # END HACK
        # HACK MLE
        # mle, _ = torch.max(probits,dim=2)
        # print(f'mle shape {mle.shape}')
        # wer = sb.utils.edit_distance.accumulatable_wer_stats(
        #     tokens, mle
        # )
        # loss = wer['WER']
        # END HACK
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
        try:
            if self.check_gradients(loss):
                self.optimizer.step()
            else:
                logger.warning(f"Bad Grads! Batch: {batch.id}")
        except ValueError as e:
            logger.warning(f"Bad Grads! Batch: {batch.id}")
            logger.error(e)
            raise e
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
