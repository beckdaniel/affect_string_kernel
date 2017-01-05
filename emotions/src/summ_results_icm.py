import numpy as np
import json
import os
import sys

RESULTS_DIR = sys.argv[1]
MODELS = ['mat32_iso', 'mat52_iso', 'mat32_iso_bias', 'mat52_iso_bias']
RANKS = ['rank_1']
SCALES = ['none']
FOLDS = [str(i) for i in range(1)]
EMOS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

for model in MODELS:
    for scale in SCALES:
        for rank in RANKS:
            emo_maes = []
            emo_rmses = []
            emo_pearsons = []
            emo_nlpd = []
            for fold in FOLDS:
                with open(os.path.join(RESULTS_DIR, model, scale, rank, fold, 'info.json')) as f:
                    info = json.load(f)
                    for emo in EMOS:
                        maes = []
                        rmses = []
                        pearsons = []
                        nlpds = []
                        maes.append(info[emo]['mae'])
                        rmses.append(info[emo]['rmse'])
                        pearsons.append(info[emo]['pearsonr'][0])
                        nlpds.append(info[emo]['nlpd'])
                        #print model, ard, scale, emo
                        #print np.mean(maes), np.mean(rmses), np.mean(pearsons),
                emo_maes.append(np.mean(maes))
                emo_rmses.append(np.mean(rmses))
                emo_pearsons.append(np.mean(pearsons))
                emo_nlpd.append(np.mean(nlpds))
            print model, rank
            print np.mean(emo_maes), np.mean(emo_rmses), np.mean(emo_pearsons),
            if emo_nlpd != []:
                print np.mean(emo_nlpd)
            else:
                print ''
    
