""" Creates test corpora to compare stratification models.
    i.e. Creates a wide and tall test set as the concatenation
         of all trials with random seeds. Then copies this 
        into the evaluation input folders and removes rows
        that were sampled into the train and dev sets on a per-seed basis.
"""
import pandas as pd
import os
import re

wide_tests = []
tall_tests = []
for (root, dirs, fns) in os.walk('.'):
    for fn in fns:
        fpath = os.path.join(root, fn)
        if 'med_wide' in fpath and fn == 'test.csv':
            wide_tests.append(pd.read_csv(fpath))
        if 'med_tall' in fpath and fn == 'test.csv':
            tall_tests.append(pd.read_csv(fpath))

wide_test = pd.concat(wide_tests).groupby('wav').sample(n=1)
tall_test = pd.concat(tall_tests).groupby('wav').sample(n=1)
print(f"wide.test = {len(wide_test)}")
print(f"tall.test = {len(tall_test)}")
            
def make_tests(rootdir, roottests):
    for (root, dirs, fns) in os.walk(rootdir):
        for d in dirs:
            if d == 'save' or 'CKPT' in d or 'save' in root:
                continue
            train = pd.read_csv(os.path.join(root,d,'train.csv'))
            dev = pd.read_csv(os.path.join(root,d,'dev.csv'))
            seed_test = roottests[~roottests['wav'].isin(train['wav']) & \
                                  ~roottests['wav'].isin(dev['wav'])]
            minid = 0
            maxid = len(train)
            train = train.assign(ID = list(range(minid, maxid)))
            minid = maxid + 1
            maxid = minid + len(dev)
            dev = dev.assign(ID = list(range(minid, maxid)))
            minid = maxid + 1
            maxid = minid + len(seed_test)
            seed_test = seed_test.assign(ID = list(range(minid, maxid)))
            train.to_csv(os.path.join(root, d, 'train.csv'))
            dev.to_csv(os.path.join(root, d, 'dev.csv'))
            seed_test.to_csv(os.path.join(root, d, 'test.csv'))
    
make_tests('tall_vs_wide', wide_test)
make_tests('wide_vs_tall', tall_test)
