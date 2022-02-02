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


def reconstruct_file_structure():
    """
    reformat the structure of all related files
    """
    pass