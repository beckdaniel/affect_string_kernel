import numpy as np
import json
import os
import sys

RESULTS_DIR = sys.argv[1]
FOLDS = [str(i) for i in range(10)]
EMOS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
#EMOS = ['anger']
#EMOS = ['joy']
#EMOS = ['disgust']
#EMOS = ['surprise']
model = 'sk_bias'

emo_noises = []
emo_biases = []
emo_gapds = []
emo_matchds = []
emo_coefs1 = []
emo_coefs2 = []
emo_coefs3 = []
emo_coefs4 = []
emo_coefs5 = []
for emo in EMOS:
    noises = []
    biases = []
    gapds = []
    matchds = []
    coefs1 = []
    coefs2 = []
    coefs3 = []
    coefs4 = []
    coefs5 = []
    for fold in FOLDS:
        with open(os.path.join(RESULTS_DIR, model, emo, str(fold), 'info.json')) as f:
            info = json.load(f)
            noises.append(info['Gaussian_noise.variance'])
            biases.append(info['sum.bias.variance'])
            gapds.append(info['sum.string.gap_decay'])
            matchds.append(info['sum.string.match_decay'])
            coefs1.append(info['sum.string.coefs'][0])
            coefs2.append(info['sum.string.coefs'][1])
            coefs3.append(info['sum.string.coefs'][2])
            coefs4.append(info['sum.string.coefs'][3])
            coefs5.append(info['sum.string.coefs'][4])

    print emo
    print "NOISE: " + str(np.median(noises))
    print "BIAS: " + str(np.median(biases))
    print "GAP: " + str(np.median(gapds))
    print "MATCH: " + str(np.median(matchds))
    print "COEF 1: " + str(np.median(coefs1))
    print "COEF 2: " + str(np.median(coefs2))
    print "COEF 3: " + str(np.median(coefs3))
    print "COEF 4: " + str(np.median(coefs4))
    print "COEF 5: " + str(np.median(coefs5))
