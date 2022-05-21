#!/usr/bin/env/python3
"""
Loads pre-trained model from file and evaluates test dataset
"""

import sys
import torch
import logging
from tqdm.contrib import tqdm
import speechbrain as sb
from torch.utils.data import DataLoader
from speechbrain.utils.distributed import run_on_main
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.dataloader import make_dataloader
from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.WARNING)


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
    
    # 1.  # Dataset prep 
    from seq_prepare import prepare_bpc, dataio_prepare  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_bpc,
        kwargs={
            "cluster": hparams["cluster"],
            "dataset_name": hparams['dataset_name'],
            "splits": hparams["splits"],
            "output_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data, tokenizer = dataio_prepare(hparams)
    test_loader = sb.dataio.dataloader.make_dataloader(
        test_data, looped_nominal_epoch=None, **hparams["test_dataloader_opts"]
    )

    # We download and pretrain the tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    logger.info("Loading pretrained ...")
    asr_model =  EncoderDecoderASR.from_hparams(source=hparams['pretrained_model'])
    asr_model.eval()

    avg_test_loss = 0.0
    cer_metric = hparams['cer_computer']()
    wer_metric = hparams['wer_computer']()
    with torch.no_grad():
        for batch in tqdm(test_loader, dynamic_ncols=True):
            wavs, predicted_tokens = asr_model.transcribe_batch(*batch.sig)
            predicted_words = [
                tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            if hparams.remove_spaces:
                predicted_words = ["".join(p) for p in predicted_words]
                target_words = ["".join(t) for t in target_words]
            cer_metric.append(ids, predicted_words, target_words)
            wer_metric.append(ids, predicted_words, target_words)

    stage_stats = {
        "CER" : cer_metric.summarize("error_rate"),
        "WER" : wer_metric.summarize("error_rate")
    }
    hparams.train_logger.log_stats(test_stats=stage_stats)
    with open(hparams.cer_file, "w") as w:
        cer_metric.write_stats(w)
    with open(hparams.wer_file, "w") as w:
        wer_metric.write_stats(w)
