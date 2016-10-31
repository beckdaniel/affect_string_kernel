import numpy as np
import json
import os
import sys

RESULTS_DIR = sys.argv[1]
#MODELS = ['ridge', 'svr', 'rbf', 'mat32', 'mat52', 'ratquad', 'linear', 'mlp']
MODELS = ['rbf', 'mat32', 'mat52', 'ratquad', 'linear', 'mlp']
#MODELS = ['mat32', 'mat52', 'ratquad', 'linear', 'mlp']
#MODELS = ['ratquad', 'linear', 'mlp']
#MODELS = ['ridge']
#ARDS = ['iso']#, 'ard']
ARDS = ['ard']
SCALES = ['none', 'scale', 'warp']
#SCALES = ['none']
#SCALES = ['scale']
#SCALES = ['warp']
FOLDS = [str(i) for i in range(10)]
EMOS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
#EMOS = ['anger']
#EMOS = ['joy']
#EMOS = ['disgust']

for model in MODELS:
    for scale in SCALES:
        for ard in ARDS:
            emo_maes = []
            emo_rmses = []
            emo_pearsons = []
            emo_nlpd = []
            for emo in EMOS:
                maes = []
                rmses = []
                pearsons = []
                nlpds = []
                for fold in FOLDS:
                    with open(os.path.join(RESULTS_DIR, model + '_' + ard, scale, emo, fold, 'info.json')) as f:
                        info = json.load(f)
                        maes.append(info['mae'])
                        rmses.append(info['rmse'])
                        pearsons.append(info['pearsonr'][0])
                        if 'nlpd' in info:
                            nlpds.append(info['nlpd'])
                #print model, ard, scale, emo
                #print np.mean(maes), np.mean(rmses), np.mean(pearsons),
                emo_maes.append(np.mean(maes))
                emo_rmses.append(np.mean(rmses))
                emo_pearsons.append(np.mean(pearsons))
                if nlpds != []:
                    #print np.mean(nlpds)
                    emo_nlpd.append(np.mean(nlpds))
                #else:
                #    print ''
            print model, ard, scale
            print np.mean(emo_maes), np.mean(emo_rmses), np.mean(emo_pearsons),
            if emo_nlpd != []:
                print np.mean(emo_nlpd)
            else:
                print ''
    
