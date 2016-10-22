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
from scipy.stats.stats import pearsonr

import GPy
import util



############
# Parse args

parser = argparse.ArgumentParser()
parser.add_argument('inputs', help='the file with the sentence inputs')
parser.add_argument('labels', help='the file with the valence labels')
parser.add_argument('embs', help= 'the word embeddings file')
parser.add_argument('model', help='one of: "ridge", "svr", "linear",' +
                    '"rbf", "mat32", "mat52", "ratquad"')
parser.add_argument('label_preproc', help='one of "none", "scale", "warp"')
parser.add_argument('output_dir', help='directory where outputs will be stored')
parser.add_argument('--data_size', help='size of dataset, default is full size',
                    default=10000, type=int)
args = parser.parse_args()

###########
# Load data

embs = util.load_embs(args.embs)
X = []
with open(args.inputs) as f:
    for line in f:
        X.append(util.preprocess_sent(line.split('_')[1]))
Y = np.loadtxt(args.labels)[:, 1:]

###############
# Preprocessing
X = np.array([util.average_sent(sent, embs) for sent in X])
data = np.concatenate((X, Y), axis = 1)[:args.data_size]
np.random.seed(1000)
np.random.shuffle(data)
X = data[:, :-1]
Y = data[:, -1:]

##############
# Get folds
kf = KFold(n_splits=10)
folds = kf.split(data)

##############
# Create output structure
main_out_dir = os.path.join(args.output_dir, 'avg', args.model, args.label_preproc)

#############
# Train models and report
fold = 0
for i_train, i_test in folds:
    X_train = X[i_train]
    Y_train = Y[i_train]
    X_test = X[i_test]
    Y_test = Y[i_test]
    
    # Select and train model
    if args.model == 'ridge':
        model = RidgeCV(alphas=np.logspace(-2, 2, 5))
        model.fit(X_train, Y_train.flatten())
    elif args.model == 'svr':
        hypers = {'C': np.logspace(-2, 2, 5),
                  'epsilon': np.logspace(-3, 1, 5),
                  'gamma': np.logspace(-3, 1, 5)}
        model = GridSearchCV(SVR(), hypers)
        model.fit(X_train, Y_train.flatten())
    elif args.model == 'rbf':
        k = GPy.kern.RBF(X.shape[1])
        model = GPy.models.GPRegression(X_train, Y_train, kernel=k)
        model.optimize(verbose=True, max_iters=50)
    
    # Get predictions
    info_dict = {}
    if args.model == 'ridge' or args.model == 'svr':
        preds = model.predict(X_test)
        info_dict['mae'] = MAE(preds, Y_test.flatten())
        info_dict['rmse'] = np.sqrt(MSE(preds, Y_test.flatten()))
        info_dict['pearsonr'] = pearsonr(preds, Y_test.flatten())
    else:
        preds, vars = model.predict(X_test)
        info_dict['mae'] = MAE(preds, Y_test)
        info_dict['rmse'] = np.sqrt(MSE(preds, Y_test))
        info_dict['pearsonr'] = pearsonr(preds.flatten(), Y_test.flatten())

    # Get parameters
    if args.model == 'ridge':
        info_dict['coefs'] = list(model.coef_)
        info_dict['intercept'] = model.intercept_
        info_dict['regularization'] = model.alpha_
    elif args.model == 'svr':
        info_dict['regularization'] = model.best_params_['C']
        info_dict['epsilon'] = model.best_params_['epsilon']
        info_dict['gamma'] = model.best_params_['gamma']
    elif args.model == 'rbf'
    
    # Save information
    fold_dir = os.path.join(main_out_dir, str(fold))
    try:
        os.makedirs(fold_dir)
    except OSError:
        # Already exists
        pass
    with open(os.path.join(fold_dir, 'info.json'), 'w') as f:
        json.dump(info_dict, f)
    np.savetxt(os.path.join(fold_dir, 'preds.tsv'), preds)

    # Next fold
    fold += 1
