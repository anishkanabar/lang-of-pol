"""
format corpus to pyannote readable structure
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

from pyannote.database import get_protocol, FileFinder


def load_data(corpus='ami'):
    """
    load dataset
    @corpus: the name of corpus 
    """
    preprocessors = {'audio': FileFinder()}
    protocol = get_protocol(f"{corpus.upper()}.SpeakerDiarization.only_words", preprocessors=preprocessors)
    return protocol


def generate_rttm():
    """
    according to the corpus structure to generate corresponding rttm files
    """
    pass


def generate_uem():
    """
    according to the corpus structure to generate corresponding uem files
    """
    pass


def generate_lst():
    """
    according to the corpus structure to generate correspinding lst files
    """
    pass