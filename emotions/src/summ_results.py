import numpy as np
import json
import os
import sys

RESULTS_DIR = sys.argv[1]
MODELS = ['ridge', 'svr', 'rbf', 'mat32', 'mat52', 'ratquad', 'linear', 'mlp']
#MODELS = ['ridge', 'rbf', 'mat32', 'mat52', 'ratquad', 'linear', 'mlp']
#MODELS = ['rbf', 'mat32', 'mat52', 'ratquad', 'linear', 'mlp']
#MODELS = ['mat32', 'mat52', 'ratquad', 'linear', 'mlp']
#MODELS = ['ratquad', 'linear', 'mlp']
MODELS = ['ridge', 'svr', 'linear', 'rbf', 'mat32', 'mat52']
#MODELS = ['ridge']
#MODELS = ['svr']
#MODELS = ['rbf']
#MODELS = ['mat32']
#MODELS = ['ridge', 'svr', 'rbf', 'mat32', 'mat52']
RANKS = ['rank_1']
#ARDS = ['iso', 'ard']
#ARDS = ['ard']
ARDS = ['iso']
#SCALES = ['none', 'scale', 'warp']
#SCALES = ['none', 'warp']
#BIAS = ['', '_bias']
BIAS = ['']
#BIAS = ['_bias']
#NORMS = ['', '_norm']
NORMS = ['']
SCALES = ['none']
#SCALES = ['scale']
#SCALES = ['warp']
FOLDS = [str(i) for i in range(10)]
EMOS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
#EMOS = ['anger']
#EMOS = ['joy']
#EMOS = ['disgust']
#EMOS = ['surprise']

for model in MODELS:
    for scale in SCALES:
        for ard in ARDS:
            for bias in BIAS:
                if model == 'svr' or model == 'ridge':
                    if scale == 'warp':
                        continue
                    if bias == '_bias':
                        continue

                for norm in NORMS:
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
                            with open(os.path.join(RESULTS_DIR, model + '_' + ard + bias + norm, scale, emo, fold, 'info.json')) as f:
                                info = json.load(f)
                                maes.append(info['mae'])
                                rmses.append(info['rmse'])
                                pearsons.append(info['pearsonr'][0])
                                if 'nlpd' in info:
                                    nlpds.append(info['nlpd'])
                        print model, ard, scale, emo
                        print "%.3f & %.3f & %.3f & %.3f" % (np.mean(nlpds), np.mean(maes), np.mean(pearsons), np.mean(rmses))
                        #print np.median(maes), np.median(rmses), np.median(pearsons), np.median(nlpds)
                        emo_maes.append(np.mean(maes))
                        emo_rmses.append(np.mean(rmses))
                        emo_pearsons.append(np.mean(pearsons))
                        #emo_maes.append(np.median(maes))
                        #emo_rmses.append(np.median(rmses))
                        #emo_pearsons.append(np.median(pearsons))
                        if nlpds != []:
                            #print np.mean(nlpds)
                            #emo_nlpd.append(np.median(nlpds))
                            emo_nlpd.append(np.mean(nlpds))
                                #else:
                                #    print ''
                    print model, ard, scale, bias, norm
                    print "%.3f & %.3f & %.3f & %.3f" % (np.mean(emo_nlpd), np.mean(emo_maes), np.mean(emo_pearsons), np.mean(emo_rmses))
                    #print np.mean(emo_maes), np.mean(emo_rmses), np.mean(emo_pearsons),
                    #if emo_nlpd != []:
                    #    print np.mean(emo_nlpd)
                    #else:
                    #    print ''
    
