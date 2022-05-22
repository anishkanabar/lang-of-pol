import logging
import torch
import pandas as pd
import speechbrain as sb
import corpus_pipeline as pipeline
from enum import Enum, auto
from asr_dataset.base import AsrETL
from asr_dataset.police import BpcETL

logger = logging.getLogger('asr.prepare.unk')


class DefineUncertain(Enum):
    BRACKETS = auto()
    PERFECT = auto()
    DISTANCE = auto()
    INAUDIBLE = auto()


def dataio_prepare(hparams):
    """ Dataset transformation pipeline """

    tokenizer = hparams["tokenizer"]

    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    train_data, val_data, test_data = pipeline.dataio_prepare(hparams, text_pipeline)
    return train_data, val_data, test_data, tokenizer
    

def create_manifests(splits, output_folder, unk_def, **kwargs):
    pipeline.create_basic_manifests(
        splits=splits, 
        output_folder=output_folder,
        filter_uncertain=False, 
        **kwargs)

    splitdata = pipeline.get_splits(splits, output_folder)
    splitdata = pipeline.get_splits(splits, output_folder)
    splitdata = {k: label_unk(v, unk_def) for k, v in splitdata.items()}

    for k, v in splitdata.items():
        AsrETL._describe(v, k)
    pipeline.write_splits(splitdata, output_folder)
    return splitdata


def label_unk(df: pd.DataFrame, uncertainty) -> pd.DataFrame:
    uncertain_def = DefineUncertain[uncertainty.upper()]
    if uncertain_def == DefineUncertain.PERFECT:
        raise NotImplementedError() # requires pefect agreement on aligned utt
    elif uncertain_def == DefineUncertain.DISTANCE:
        raise NotImplementedError() # requires small levenstein on aligned utt
    elif uncertain_def == DefineUncertain.BRACKETS:
        pat = '\[.+\]'
        is_unk = df['transcript'].str.contains(pat, regex=True)
        return df.assign(unk = is_unk.where(is_unk == 0, -1))
    elif uncertain_def == DefineUncertain.INAUDIBLE:
        pat = '|'.join(BpcETL.BAD_WORDS)
        is_unk = df['transcript'].str.contains(pat, regex=True, case=False) 
        return df.assign(unk = is_unk.where(is_unk == 0, -1))
