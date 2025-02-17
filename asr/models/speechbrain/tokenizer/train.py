#!/usr/bin/env/python3
"""Recipe for training a BPE tokenizer with BPC.
The tokenizer coverts transcripts into sub-word units that can
be used to train a language (LM) or an acoustic model (AM).

To run this recipe, do the following:
> python train.py hparams/tokenizer_bpe5000.yaml
"""

import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

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

    # 1.  # Dataset prep (parsing timers-and-such)
    from tok_prepare import create_manifests  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        create_manifests,
        kwargs={
            "cluster": hparams["cluster"],
            "dataset_name": hparams['dataset_name'],
            "splits": hparams["splits"],
            "output_folder": hparams["output_folder"],
        },
    )

    # Train tokenizer
    hparams["tokenizer"]()
