"""
format corpus to pyannote readable structure
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

import argparse
from pyannote.database import get_protocol, FileFinder
import os, glob
import pandas as pd


def load_data(corpus='ami', prefix='only_words'):
    """
    load dataset
    @corpus: the name of corpus 
    @prefix: the name of the subcorpus
    """
    preprocessors = {'audio': FileFinder()}
    protocol = get_protocol(f"{corpus.upper()}.SpeakerDiarization.{prefix}", preprocessors=preprocessors)
    return protocol


def make_corpus_structure(root_path, corpus_name):
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
    devlopment = all_files[len(train):len(train)+int(len(all_files)*dev_rate)]
    test = all_files[len(train)+len(devlopment):]
    # inform the spliting results
    print('Corpus splitting successfully.')
    print(f'Training sample: {len(train)}.\nDevelopment sample: {len(devlopment)}.\nTesting sample: {len(test)}.')
    return train, devlopment, test


def generate_rttms(root_path, train, development, test, reference, prefix_ignore=''):
    utt = pd.read_csv(reference)
    # train files
    for path in train:
        uri = path.split('/')[-1].split('.')[0]
        sub_df = utt[utt.filePath == path.replace(prefix_ignore, '')]
        file = open(root_path+'/rttms/train/'+uri+'.rttm', 'w')
        for index, row in sub_df.iterrows():
            try:
                file.write(f"SPEAKER {uri} 1 {row['start']} {float(row['end'])-float(row['start'])} <NA> <NA> {row['speaker'].replace(' ', '')} <NA> <NA>" + "\n")
            except BaseException:
                print(f"Incorrect training instance in .csv file row {index}.")
        file.close()
    for path in development:
        uri = path.split('/')[-1].split('.')[0]
        sub_df = utt[utt.filePath == path.replace(prefix_ignore, '')]
        file = open(root_path+'/rttms/dev/'+uri+'.rttm', 'w')
        for index, row in sub_df.iterrows():
            try:
                file.write(f"SPEAKER {uri} 1 {row['start']} {float(row['end'])-float(row['start'])} <NA> <NA> {row['speaker'].replace(' ', '')} <NA> <NA>" + "\n")
            except BaseException:
                print(f"Incorrect development instance in .csv file row {index}.")
        file.close()
    for path in test:
        uri = path.split('/')[-1].split('.')[0]
        sub_df = utt[utt.filePath == path.replace(prefix_ignore, '')]
        file = open(root_path+'/rttms/test/'+uri+'.rttm', 'w')
        for index, row in sub_df.iterrows():
            try:
                file.write(f"SPEAKER {uri} 1 {row['start']} {float(row['end'])-float(row['start'])} <NA> <NA> {row['speaker'].replace(' ', '')} <NA> <NA>" + "\n")
            except BaseException:
                print(f"Incorrect testing instance in .csv file row {index}.")
        file.close()
    print(".rttm files created successfully!")


def generate_uems(root_path, train, development, test, reference, prefix_ignore=''):
    utt = pd.read_csv(reference)
    # train files
    for path in train:
        uri = path.split('/')[-1].split('.')[0]
        sub_df = utt[utt.filePath == path.replace(prefix_ignore, '')]
        file = open(root_path+'/uems/train/'+uri+'.uem', 'w')
        file.write(f"{uri} 1 0.000 {sub_df.tail(1)['end'].values[0]}")
        file.close()
    for path in development:
        uri = path.split('/')[-1].split('.')[0]
        sub_df = utt[utt.filePath == path.replace(prefix_ignore, '')]
        file = open(root_path+'/uems/dev/'+uri+'.uem', 'w')
        file.write(f"{uri} 1 0.000 {sub_df.tail(1)['end'].values[0]}")
        file.close()
    for path in test:
        uri = path.split('/')[-1].split('.')[0]
        sub_df = utt[utt.filePath == path.replace(prefix_ignore, '')]
        file = open(root_path+'/uems/test/'+uri+'.uem', 'w')
        file.write(f"{uri} 1 0.000 {sub_df.tail(1)['end'].values[0]}")
        file.close()
    print(".uem files created successfully!")


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
            uris = [audio.split('/')[-1].split('.')[0] + '\n' for audio in data]
            f.writelines(uris)
            f.close()
    print('.lst files are created successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build new corpus scripts for pyannote.')
    parser.add_argument('corpus_name', default='ami', help='the name of the corpus (default: ami)')
    parser.add_argument('root_path', help='path of saving corpus scripts.')
    parser.add_argument('corpus_prefix', default=0, help='origional location of the corpus.')
    parser.add_argument('audio_syntax', help='syntax to all audio file path.')
    parser.add_argument('-r', '--reference_file', type=str, default='', help='the location of reference .csv file.')
    parser.add_argument('-i', '--ignore_prefix', type=str, default='', help='ignoring part of filepath.')
    parser.add_argument('-t', '--train_num', type=float, default=0.6, help='portion of training sample.')
    parser.add_argument('-d', '--dev_num', type=float, default=0.2, help='portion of development sample.')
    
    args = parser.parse_args()

    make_corpus_structure(args.root_path, args.corpus_name)
    train, dev, test = split_data(args.audio_syntax, args.train_num, args.dev_num)
    generate_list(root_path=args.root_path+args.corpus_name, train=train, development=dev, test=test)
    generate_rttms(root_path=args.root_path+args.corpus_name, train=train, development=dev, test=test,
               reference=args.reference_file, prefix_ignore=args.ignore_prefix)
    generate_uems(root_path=args.root_path+args.corpus_name, train=train, development=dev, test=test,
             reference=args.reference_file, prefix_ignore=args.ignore_prefix)
    print(f'{args.corpus_name} created successfully!')
