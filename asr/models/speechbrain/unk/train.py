#!/usr/bin/env/python3
"""
BPC seq2seq model recipe. (Adapted from the LibriSpeech recipe.)
"""

import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.WARNING)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """
        Forward computations from the waveform batches to the output probabilities.
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        x = self.modules.enc(feats.detach())
        encoding = x[:, -1, :].squeeze()
        logits = self.modules.classifier(encoding)
        return logits, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """
        Computes the loss (CTC+NLL) given predictions and targets.
        """

        current_epoch = self.hparams.epoch_counter.current
        logits, wav_lens = predictions
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss = self.hparams.classifier_cost(logits, labels)

        if stage != sb.Stage.TRAIN:
            scores = torch.where(logits > .5, 1, -1)
            self.err_metric.append(ids, scores, labels)

        return loss

    def fit_batch(self, batch):
        """
        Train the parameters given a single batch in input
        """
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.batch_idx += 1
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """
        Computations needed for validation/test batches
        """
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """
        Gets called at the beginning of each epoch
        """
        self.batch_idx = 0
        if stage != sb.Stage.TRAIN:
            self.err_metric = self.hparams.err_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """
        Gets called at the end of a epoch.
        """
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ERR"] = self.err_metric.summarize()

        # Perform end-of-iteration things, like annealing, logging, etc.
        # XXX: Think more critically about which metric to minimize
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["precision"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"precision": stage_stats["precision"]},
                min_keys=["precision"]
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.err_file, "w") as w:
                self.err_metric.write_stats(w)


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep 
    from unk_prepare import create_manifests, dataio_prepare  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        create_manifests,
        kwargs={
            "cluster": hparams["cluster"],
            "dataset_name": hparams['dataset_name'],
            "splits": hparams["splits"],
            "output_folder": hparams["output_folder"],
            "seed": hparams["seed"],
            "ambiguity_strategy": hparams["ambiguity_strategy"],
            "unk_def": hparams["unk_def"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data, tokenizer = dataio_prepare(hparams)

    # We download and pretrain the tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = tokenizer

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    asr_brain.evaluate(
        test_data, test_loader_kwargs=hparams["test_dataloader_opts"]
    )
