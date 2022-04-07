import pickle
import opensmile
import time
import tracemalloc
import argparse

def available_features():      #Displays the features which can be obtained
    features = ['Energy','Melspec','Intensity','Mfcc']
    return features

def make_config_string(feature_list,func_list):    #Makes a config file using the feature list and func list
    func_dict = {'variance':0,'stddev':0,'amean':0,'skewness':0,'kurtosis':0,'doRatioLimit':0}
    #Preparing dictionaries to be used in the future
    for key in list(func_dict.keys()):
        if key in func_list:
            func_dict[key] = 1
            
    #Preparing strings to be used
    func_str = ''
    for key in list(func_dict.keys()):
        func_str = func_str + 'Moments.{_key} = {_val}\n'.format(_key = key, _val = func_dict[key])
    
    #This part is necessary regardless of input
    config_str = '''
    [componentInstances:cComponentManager]
    instance[dataMemory].type=cDataMemory

    ;;; default source
    [componentInstances:cComponentManager]
    instance[dataMemory].type=cDataMemory

    ;;; source

    \{\cm[source{?}:include external source]}

    ;;; main section

    [componentInstances:cComponentManager]
    instance[framer].type = cFramer
    '''
    
    #instantiating the feature sections
    inst_str = ''
    for feature in feature_list:
        inst_str = inst_str + 'instance[{_feat}].type = c{_feat}\n'.format(_feat = feature)
    config_str = config_str + inst_str
    
    config_str = config_str + '''instance[func].type=cFunctionals
    
    [framer:cFramer]
    reader.dmLevel = wave
    writer.dmLevel = frames
    copyInputName = 1
    frameMode = fixed
    frameSize = 0.025000
    frameStep = 0.010000
    frameCenterSpecial = left
    noPostEOIprocessing = 1

    '''
    
    #Coding descriptions for each of the features in the input. The further arguments would need to be provided here
    desc_str = ''
    feat_str = ''
    for feature in feature_list:
        feat_str = feat_str + feature + ';'
        desc_str = desc_str + '''[{_feat}:c{_feat}]
        reader.dmLevel = frames
        writer.dmLevel = {_feat}
        '''.format(_feat = feature) + '\{\cm[bufferModeRbConf{?}:path to included config to set the buffer mode for the standard ringbuffer levels]}' + '''
        nameAppend = {_feat}
        copyInputName = 1
        '''.format(_feat=feature) + '\n'
    config_str = config_str + desc_str
    feat_str = feat_str[:-1]
    
    #Functionals to be computed for each of the features. This function is complete
    func_append = '''    [func:cFunctionals]
    reader.dmLevel=%s
    writer.dmLevel=func
    copyInputName = 1
    \{\cm[bufferModeRbConf]}
    \{\cm[frameModeFunctionalsConf{?}:path to included config to set frame mode for all functionals]}
    functionalsEnabled=Moments
    ''' % feat_str
    
    config_str = config_str + func_append + func_str
    end_str = '''

    ;;; sink

    \{\cm[sink{?}:include external sink]}

    '''
    config_str = config_str + end_str
    print(config_str)
    return config_str
    
if __name__ == 'main': #Will be called on running >python feature_extractor.py
    pkl_path = '/project/graziul/ra/ajays/whitelisted_vad_dict.pkl'
    file = open(pkl_path,'rb')
    vad_dict = pickle.load(file)
    file.close()


    parser = argparse.ArgumentParser(description='Process Args')
    parser.add_argument("--features", nargs="+", default=["Energy"])
    parser.add_argument("--functionals", nargs="+", default=["stddev","amean"])

    args = parser.parse_args()

    features_list = args.features
    func_list = args.functionals
    print(features_list, func_list)
    make_config_string(features_list,func_list)
