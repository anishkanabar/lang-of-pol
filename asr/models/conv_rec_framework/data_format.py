"""
format corpus to pyannote readable structure
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

from pyannote.database import get_protocol, FileFinder
import os


def load_data(corpus='ami'):
    """
    load dataset
    @corpus: the name of corpus 
    """
    preprocessors = {'audio': FileFinder()}
    protocol = get_protocol(f"{corpus.upper()}.SpeakerDiarization.only_words", preprocessors=preprocessors)
    return protocol


def make_corpus_strcuture(root_path, corpus_name):
    """
    construct the folders' structure of the corpus
    """
    if not os.path.exists(root_path + corpus_name):
        os.makedirs(root_path + corpus_name)
        # create three subfolders: lists, rttms, uems
        if not os.path.exists(root_path + corpus_name + '/lists'):
            os.makedirs(root_path + corpus_name + '/lists')
        else:
            print(f"Directory '{root_path + corpus_name + '/lists'}' already exist. Please delete it and try again.")
        
        if not os.path.exists(root_path + corpus_name + '/rttms'):
            os.makedirs(root_path + corpus_name + '/rttms')
        else:
            print(f"Directory '{root_path + corpus_name + '/rttms'}' already exist. Please delete it and try again.")

        if not os.path.exists(root_path + corpus_name + '/uems'):
            os.makedirs(root_path + corpus_name + '/uems')
        else:
            print(f"Directory '{root_path + corpus_name + '/uems'}' already exist. Please delete it and try again.")
        print('The structure of corpus has been set successfully.')
    else:
        print(f"Directory '{root_path + corpus_name}' already exist. Please delete it and try again.")


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