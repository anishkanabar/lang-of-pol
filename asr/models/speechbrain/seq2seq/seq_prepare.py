import logging
import torch
import speechbrain as sb
import corpus_pipeline as pipeline

logger = logging.getLogger('asr.prepare.seq')

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
    

def create_manifests(**kwargs):
    pipeline.create_manifests(text_col="wrd", **kwargs)

