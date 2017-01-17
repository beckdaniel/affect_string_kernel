"""
Models that employ an average of word embeddings as inputs.

"""
import numpy as np
import sys
import os
import argparse
import json

from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
import sklearn.preprocessing as pp
from scipy.stats.stats import pearsonr

import GPy
import util
import flakes



############
# Parse args

parser = argparse.ArgumentParser()
parser.add_argument('inputs', help='the file with the sentence inputs')
parser.add_argument('labels', help='the file with the valence labels')
parser.add_argument('embs', help= 'the word embeddings file')
#parser.add_argument('label_preproc', help='one of "none", "scale", "warp"')
parser.add_argument('output_dir', help='directory where outputs will be stored')
parser.add_argument('--data_size', help='size of dataset, default is full size',
                    default=10000, type=int)
#parser.add_argument('--ard', help='set this flag to enable ARD', action='store_true')
parser.add_argument('--bias', help='set this flag to add a bias kernel', action='store_true')
parser.add_argument('--gpu', help='set this flag to run in GPU', action='store_true')
#parser.add_argument('--norm', help='set this flag to normalise the labels', action='store_true')
args = parser.parse_args()

###########
# Constants

# This is the emotion label order in the original data.
EMOS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

###########
# Load data

embs, words = util.load_embs_matrix(args.embs)
X = []
with open(args.inputs) as f:
    for line in f:
        X.append(util.preprocess_sent(line.split('_')[1]))
Y = np.loadtxt(args.labels)[:, 1:]
#print Y

###############
# Preprocessing
X = np.array([[x] for x in X], dtype=object)
print X
print Y
data = np.concatenate((X, Y), axis=1)[:args.data_size]
np.random.seed(1000)
np.random.shuffle(data)
X = data[:, :-6]
Y = (data[:, -6:])

##############
# Get folds
kf = KFold(n_splits=10)
folds = kf.split(data)

##############
# Create output structure
mode = 'sk'
if args.bias:
    mode += '_bias'
main_out_dir = os.path.join(args.output_dir, 'sk', 'ind',  mode)

if args.gpu:
    device = '/gpu:0'
else:
    device = '/cpu:0'

#############
# Train models and report
fold = 0
for i_train, i_test in folds:
    X_train = X[i_train]
    Y_train_all = Y[i_train]
    X_test = X[i_test]
    Y_test_all = Y[i_test]
    #print Y_test_all

    for emo_id, emo in enumerate(EMOS):
        Y_train = Y_train_all[:, emo_id:emo_id+1]
        Y_test = Y_test_all[:, emo_id:emo_id+1]

        k = flakes.wrappers.gpy.GPyStringKernel(order_coefs=[1.0] * 5, embs=embs, index=words, device=device)
                
        if args.bias:
            k = k + GPy.kern.Bias(X.shape[1])
        model = GPy.models.GPRegression(X_train, Y_train, kernel=k)
        model.optimize(messages=True, max_iters=100)
    
        # Get predictions
        info_dict = {}
        preds, vars = model.predict(X_test)
        print preds, vars
        info_dict['mae'] = MAE(preds, Y_test)
        info_dict['rmse'] = np.sqrt(MSE(preds, Y_test))
        info_dict['pearsonr'] = pearsonr(preds.flatten(), Y_test.flatten())
        nlpd = -model.log_predictive_density(X_test, Y_test)
        info_dict['nlpd'] = np.mean(nlpd)

        # Get parameters

        param_names = model.parameter_names()
        for p_name in param_names:
            if p_name == 'warp_tanh.psi':
                info_dict[p_name] = list([list(pars) for pars in model[p_name]])
            else:
                try:
                    info_dict[p_name] = float(model[p_name])
                except TypeError: #ARD
                    info_dict[p_name] = list(model[p_name])
        info_dict['log_likelihood'] = float(model.log_likelihood())
    
        # Save information
        fold_dir = os.path.join(main_out_dir, emo, str(fold))
        try:
            os.makedirs(fold_dir)
        except OSError:
            # Already exists
            pass
        with open(os.path.join(fold_dir, 'info.json'), 'w') as f:
            json.dump(info_dict, f, indent=2)
        np.savetxt(os.path.join(fold_dir, 'preds.tsv'), preds)
        np.savetxt(os.path.join(fold_dir, 'vars.tsv'), vars)

    # Finished emotions, next fold
    fold += 1
