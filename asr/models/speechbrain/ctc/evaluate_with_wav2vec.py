#!/usr/bin/env/python3
"""Recipe for evaluating a trained wav2vec-based ctc ASR system.
The system employs wav2vec as its encoder. Decoding is performed with
ctc greedy decoder.
To run this recipe, do the following:
> python evaluate_with_wav2vec.py hparams/XXX.yaml

Authors
 * Eric Chandler 2022
 * Sung-Lin Yeh 2021
 * Titouan Parcollet 2021
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
"""

import os
import sys
import torch
import logging
import librosa
import pandas as pd
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from train_with_wav2vec import ASR

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.WARNING)


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
    from ctc_prepare import prepare_bpc, dataio_prepare  # noqa

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
            "skip_prep": hparams["skip_prep"],
            "seed": hparams["seed"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets, label_encoder = dataio_prepare(
        hparams
    )

    # Model initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    logger.info(f'Loading checkpoint from {asr_brain.checkpointer.checkpoints_dir}')
    ckpt = asr_brain.checkpointer.recover_if_possible(device=torch.device(asr_brain.device))
    if ckpt is None:
        raise RuntimeError('Could not find model checkpoint')

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = label_encoder
    
    # Testing
    devices = [torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())]
    logger.info(f'Running on devices {";".join(devices)}')

    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        asr_brain.evaluate(
            test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
        )
