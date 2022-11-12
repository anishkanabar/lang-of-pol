import logging
import corpus_pipeline as pipeline

logger = logging.getLogger('asr.prepare.tokenizer')

def dataio_prepare(hparams):
    """ Dataset transformation pipeline """
    return pipeline.dataio_prepare(hparams)


def create_manifests(**kwargs):
    pipeline.create_manifests(text_col='transcript', **kwargs)
