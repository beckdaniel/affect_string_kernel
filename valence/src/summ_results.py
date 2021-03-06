import numpy as np
import json
import os
import sys

RESULTS_DIR = sys.argv[1]
#MODELS = ['ridge', 'svr', 'rbf', 'mat32', 'mat52', 'ratquad', 'linear', 'mlp']
#MODELS = ['rbf', 'mat32', 'mat52', 'ratquad', 'linear', 'mlp']
#MODELS = ['mat32', 'mat52', 'ratquad', 'linear', 'mlp']
MODELS = ['ratquad', 'linear', 'mlp']
ARDS = ['iso']#, 'ard']
SCALES = ['none', 'scale', 'warp']
#SCALES = ['none']
#SCALES = ['scale']
#SCALES = ['none', 'warp']
FOLDS = [str(i) for i in range(10)]

for model in MODELS:
    for scale in SCALES:
        for ard in ARDS:
            maes = []
            rmses = []
            pearsons = []
            nlpds = []
            for fold in FOLDS:
                with open(os.path.join(RESULTS_DIR, model + '_' + ard, scale, fold, 'info.json')) as f:
                    info = json.load(f)
                    maes.append(info['mae'])
                    rmses.append(info['rmse'])
                    pearsons.append(info['pearsonr'][0])
                    if 'nlpd' in info:
                        nlpds.append(info['nlpd'])
            print model, ard, scale
            print np.mean(maes), np.mean(rmses), np.mean(pearsons),
            if nlpds != []:
                print np.mean(nlpds)
            else:
                print ''
    
