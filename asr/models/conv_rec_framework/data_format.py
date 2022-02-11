"""
format corpus to pyannote readable structure
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

from pyannote.database import get_protocol, FileFinder
import os, glob


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
    @root_path: the root path of the dataset
    @corpus_name: the name of the corpus
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
            os.makedirs(root_path + corpus_name + '/rttms/train')
            os.makedirs(root_path + corpus_name + '/rttms/dev')
            os.makedirs(root_path + corpus_name + '/rttms/test')
        else:
            print(f"Directory '{root_path + corpus_name + '/rttms'}' already exist. Please delete it and try again.")

        if not os.path.exists(root_path + corpus_name + '/uems'):
            os.makedirs(root_path + corpus_name + '/uems')
            os.makedirs(root_path + corpus_name + '/uems/train')
            os.makedirs(root_path + corpus_name + '/uems/dev')
            os.makedirs(root_path + corpus_name + '/uems/test')
        else:
            print(f"Directory '{root_path + corpus_name + '/uems'}' already exist. Please delete it and try again.")

        print('The structure of corpus has been set successfully.')
    else:
        print(f"Directory '{root_path + corpus_name}' already exist. Please delete it and try again.")


def split_data(syntax, train_rate=0.6, dev_rate=0.2):
    """
    get the location of all the audio files and split them to training, development, and test samples
    @train_rate: the porportion of training sample
    @dev_rate: the porportion of development sample
    """
    all_files = glob.glob(syntax)
    all_files = [f for f in all_files if f.endswith('.sph') or f.endswith('.wav')]
    # random sampling
    train = all_files[:int(len(all_files)*train_rate)]
    devlopment = all_files[len(train):len(train)+int(len(all_files)*0.2)]
    test = all_files[len(train)+len(devlopment):]
    # inform the spliting results
    print('Corpus splitting successfully.')
    print(f'Training sample: {len(train)}.\nDevelopment sample: {len(devlopment)}.\nTesting sample: {len(test)}.')
    return train, devlopment, test


def generate_rttm(root_path, train, development, test, reference):
    """
    according to the corpus structure to generate corresponding rttm files
    @root_path:
    @train:
    @development:
    @test:
    @reference:
    """
    pass


def generate_uem():
    """
    according to the corpus structure to generate corresponding uem files
    """
    pass


def generate_list(root_path, train, development, test):
    """
    according to the corpus structure to generate correspinding list files
    @root_path: the root path of the database
    @train: the training samples
    @development: the development samples
    @test: the test samples
    """
    for name, data in zip(['train.txt', 'dev.txt', 'test.txt'], [train, development, test]):
        with open(root_path + '/lists/' + name, 'w') as f:
            uris = [audio.split('/')[-1].split('.')[0] for audio in data]
    print('.lst files are created successfully.')