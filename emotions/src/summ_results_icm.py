import numpy as np
import json
import os
import sys

RESULTS_DIR = sys.argv[1]
#MODELS = ['mat32_iso', 'mat52_iso', 'mat32_iso_bias', 'mat52_iso_bias']
MODELS = ['mat32_iso_bias', 'mat52_iso_bias']
#MODELS = ['mat32_iso_bias']
RANKS = ['rank_1', 'rank_2', 'rank_3', 'rank_4', 'rank_5']
#RANKS = ['rank_1']
SCALES = ['none']
FOLDS = [str(i) for i in range(10)]
EMOS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

for model in MODELS:
    for scale in SCALES:
        for rank in RANKS:
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
                    with open(os.path.join(RESULTS_DIR, model, scale, rank, fold, 'info.json')) as f:
                        info = json.load(f)
                
                    maes.append(info[emo]['mae'])
                    rmses.append(info[emo]['rmse'])
                    pearsons.append(info[emo]['pearsonr'][0])
                    nlpds.append(info[emo]['nlpd'])
                #print model, rank, emo                
                #print np.mean(maes), np.mean(rmses), np.mean(pearsons), np.mean(nlpds)
                #print np.median(maes), np.median(rmses), np.median(pearsons), np.median(nlpds)
                emo_maes.append(np.mean(maes))
                emo_rmses.append(np.mean(rmses))
                emo_pearsons.append(np.mean(pearsons))
                emo_nlpd.append(np.mean(nlpds))
                #emo_maes.append(np.median(maes))
                #emo_rmses.append(np.median(rmses))
                #emo_pearsons.append(np.median(pearsons))
                #emo_nlpd.append(np.median(nlpds))
            print model, rank, 'TOTAL'
            print np.mean(emo_maes), np.mean(emo_rmses), np.mean(emo_pearsons),
            if emo_nlpd != []:
                print np.mean(emo_nlpd)
            else:
                print ''
    
