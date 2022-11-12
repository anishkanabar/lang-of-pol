import argparse
import numpy as np
import pandas as pd
import kenlm
import subprocess
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")
import time
import nltk as nltk
# nltk.download('punkt')
from nltk.tokenize import ToktokTokenizer


'''Dependencies: boost and nltk.download('punkt') already installed in environment'''


#defining functions

def kenlm_model_test(binary_file_loc, test_df):
    ''' 
    Estimate KenLM log probability and perplexity scores from KenLM arpa or binary of arpa
    Arguments:
     1) binary_file_loc:  binary file which was built from an arpa file 
     2) test_df: dataframe containing test transcript sentences
    
    Returns modified dataframe with probability and perplexity scores for each sentence
    '''
    #fit kenlm model
    model = kenlm.Model(binary_file_loc)
    
    test_df['score'] = np.NaN
    test_df['perplexity'] = np.NaN
    
    test_df['score'] = test_df['cleaned_transcription'].apply(lambda x: model.score(x))
    test_df['perplexity'] = test_df['cleaned_transcription'].apply(lambda x: model.perplexity(x))
    
    return test_df



def build_arpa_binary(klm_loc,n_gram, train_file_path, output_folder):
    '''
    Builds arpa file from corpus text file and then builds binary file
    Arguments:
    1) klm_loc: folder location of KenLM bin folder
    2) n_gram: the n_gram order of the arpa file
    3) train_file_path: full location of the 'train' corpus text file
    4) output_folder: output folder
    Creates:
    1) arpa file
    2) binary file
    Returns:
    Location of the arpa binary file created
    '''
    #variable initializations
    #Defining commands and arguments in shell syntax
    ##command to create the arpa file
    ##input
    input1 = "<" + train_file_path
    ##outputs
    arpa_file = 'klm_' + str(n_gram) + 'gram' + '.arpa'
    output1 = '>' + output_folder + arpa_file
    ##other arguments
    fallback = '--discount_fallback'    #argument for small train datasets else get error #ATC0 dataset needed this
    argT1 = '-T' #argument recommended by KenLM dev; to explicitly mention location of a temp folder
    argT2 = output_folder + 'tmp/'
    # argT2 = '/home/ubuntu/klm_perplexity/data/tmp/'  #location of the temp folder in ubuntu #DELETE
    

    #KenLM command
    lmplz = klm_loc + 'lmplz'

    # cmd_list = [lmplz, '-o', n_gram, argT1, argT2, input1, output1]
    cmd_list = [lmplz, '-o', n_gram, fallback, argT1, argT2, input1, output1] #with fallback argument
    build_arpa_cmd = ' '.join(cmd_list) #adding space between commands
    print('Assembled lm_cmd:', build_arpa_cmd)  #for debugging
    
    #run subprocess in shell to create arpa file
    arpa_result = subprocess.run(build_arpa_cmd, shell=True)
    print('Shell subprocess results:',arpa_result)
    
    ##command to build binary from arpa file
    build_bin = klm_loc + 'build_binary'
    input2 = output_folder + arpa_file
    binary_file = 'klm_' + str(n_gram) + 'gram' + '.binary'
    output2 = output_folder + binary_file
    cmd_list2 = [build_bin, input2, output2]
    build_bin_cmd = ' '.join(cmd_list2) #adding space between commands
    print('Assembled binary_cmd:', build_bin_cmd) #for debugging
    
    bin_result = subprocess.run(build_bin_cmd, shell=True)
    print('Shell subprocess run results:',bin_result)
    
    return output2

def corpus_stats(df_column):
    '''Take in a df column(Series object) and return number of sentences/rows, number of tokens and unique tokens as a list'''
    sents_count = len(df_column)
    text = ' '.join(df_column)
    tokens = nltk.word_tokenize(text)
    tkn_count =  len(tokens)
    unq_tkn_count =  len(set(tokens))

    return [sents_count, tkn_count, unq_tkn_count]



#########################################################################
#########################################################################


def main():
    
    #parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('klmLocation', help='kenlm/bin folder path')
    parser.add_argument('ngram',       help= 'n_gram order of arpa file to be built')
    parser.add_argument('inputCsv',    help= 'Location of input csv file with transcription')
    parser.add_argument('outputPath',  help='Folder location to create output files')
    args = parser.parse_args()
    klm_loc = args.klmLocation
    n_gram = str(args.ngram)
    input_csv = args.inputCsv
    output_folder = args.outputPath

    #read input csv file
    input_df = pd.read_csv(input_csv)
    data_df = input_df[['cleaned_transcription']] #isolate the transcription

    #variable initializations
    results_dict = {ky:[] for ky in ['kfold_index', 'n_gram',  'mean_log_prob', 'mean_perplexity','train_stats','test_stats']} #dictionary to capture mean results
    test_scores_dict = {}   #dictionary to capture test sentences and scores
    excel_file = 'klm_' + str(n_gram) + 'gram_scores' + '.xlsx' #excel file to save each test sentence's scores

    i = 0 #index to track each kfold run

    # instantiate kfold
    #Arguments: shuffle ensures sentences in each fold are random and random_state ensures the order of sentences in each fold is random
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    for train_idx, test_idx in kfold.split(data_df):
        
        print(f'KFOLD RUN:::::::::{i}')   #debug statement
        start_time = time.time()

        #create text file with k-1 splits created leaving only hold out data
        df_train = data_df.iloc[train_idx,:]
        df_test = data_df.iloc[test_idx,:]

        #write train sentences to txt file
        text_file_path = output_folder + 'train.txt'
        f = open(text_file_path,'w+')
        for idx,row in df_train.iterrows():
            if row['cleaned_transcription']:
                f.write(row['cleaned_transcription'])
                f.write('\n')
        f.close()

        #call function to build arpa and binary file and return location
        binary_file_loc = build_arpa_binary(klm_loc, n_gram, text_file_path, output_folder)

        build_time = round(time.time() - start_time)
        print(f'KFOLD{i} build time(seconds): {build_time}')

        #call function to fit kenlm model to binary file, find scores for test and return dataframe with results
        results_df = kenlm_model_test(binary_file_loc, df_test)

        #save each test sentence and scores 
        test_scores_dict[i] = results_df.to_dict()

        #calculate averages for current kfold run and save results
        results_dict['kfold_index'].append(i)
        results_dict['n_gram'].append(n_gram)
        results_dict['mean_log_prob'].append(results_df['score'].mean())
        results_dict['mean_perplexity'].append(results_df['perplexity'].mean())
        results_dict['train_stats'].append(corpus_stats(df_train['cleaned_transcription']))
        results_dict['test_stats'].append(corpus_stats(df_test['cleaned_transcription']))

        
        i += 1  #tracking each kfold run

    df_cv_results = pd.DataFrame.from_dict(results_dict)
    print(df_cv_results.head())

    #write all results to an excel file
    output_xlsx = output_folder + excel_file
    with pd.ExcelWriter(output_xlsx) as writer:
        df_cv_results.to_excel(writer, sheet_name = 'kfold_mean_results')

        for i in range(len(test_scores_dict)):
            df = pd.DataFrame.from_dict(test_scores_dict[i])
            sheet = 'sheet' + str(i)
            df.to_excel(writer, sheet_name = sheet)


if __name__ == "__main__":
    main()

