import pickle
import opensmile
import time
import tracemalloc
import argparse

def make_config_string(lld_list,func_list):
    lld_str = ''
    for lld in lld_list:
        lld_str = lld_str + '{_lld} = 1\n'.format(_lld = lld)
    print(lld_str)
    
    func_dict = {'variance':0,'stddev':0,'amean':0,'skewness':0,'kurtosis':0,'doRatioLimit':0}
    for key in list(func_dict.keys()):
        if key in func_list:
            func_dict[key] = 1
            
    func_str = ''
    for key in list(func_dict.keys()):
        func_str = func_str + 'Moments.{_key} = {_val}\n'.format(_key = key, _val = func_dict[key])
    print(func_str)
    

    str1 = '''
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
    instance[lld].type = cEnergy
    instance[func].type=cFunctionals

    [framer:cFramer]
    reader.dmLevel = wave
    writer.dmLevel = frames
    copyInputName = 1
    frameMode = fixed
    frameSize = 0.025000
    frameStep = 0.010000
    frameCenterSpecial = left
    noPostEOIprocessing = 1

    [lld:cEnergy]
    reader.dmLevel = frames
    writer.dmLevel = lld
    \{\cm[bufferModeRbConf{?}:path to included config to set the buffer mode for the standard ringbuffer levels]}
    nameAppend = energy
    copyInputName = 1
    '''

    str2 = '''[func:cFunctionals]
    reader.dmLevel=lld
    writer.dmLevel=func
    copyInputName = 1
    \{\cm[bufferModeRbConf]}
    \{\cm[frameModeFunctionalsConf{?}:path to included config to set frame mode for all functionals]}
    functionalsEnabled=Moments
    '''

    str3 = '''

    ;;; sink

    \{\cm[sink{?}:include external sink]}

    '''
    config_str = str1 + lld_str + str2 + func_str + str3
    return config_str

if __name__ == 'main':
    pkl_path = '/project/graziul/ra/ajays/whitelisted_vad_dict.pkl'
    file = open(pkl_path,'rb')
    vad_dict = pickle.load(file)
    file.close()


    parser = argparse.ArgumentParser(description='Process Args')
    parser.add_argument("--lld", nargs="+", default=["rms", "log"])
    parser.add_argument("--functionals", nargs="+", default=["stddev","amean"])

    args = parser.parse_args()

    lld_list = args.lld
    func_list = args.functionals
    print(lld_list, func_list)
    make_config_string(lld_list,func_list)
