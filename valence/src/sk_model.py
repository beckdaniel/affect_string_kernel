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
#parser.add_argument('model', help='one of: "ridge", "svr", "linear",' +
#                    '"rbf", "mat32", "mat52", "ratquad", "mlp"')
parser.add_argument('label_preproc', help='one of "none", "scale", "warp"')
parser.add_argument('output_dir', help='directory where outputs will be stored')
parser.add_argument('--data_size', help='size of dataset, default is full size',
                    default=10000, type=int)
#parser.add_argument('--ard', help='set this flag to enable ARD', action='store_true')
args = parser.parse_args()

###################
# Load data

embs, index = util.load_embs_matrix(args.embs)
X = []
with open(args.inputs) as f:
    for line in f:
        sent = util.preprocess_sent(line.split('_')[1])
        indices = util.get_indices(sent, index)
        X.append([indices])
Y = np.loadtxt(args.labels)[:, 1:]

###################
# Preprocessing

#X = util.pad_sents(X)
#print X[:10]
#print Y[:10]
data = np.concatenate((X, Y), axis=1)[:args.data_size]
np.random.seed(1000)
np.random.shuffle(data)
X = data[:, :-1]
Y = data[:, -1:]
#print data
#print X
#print Y

##############
# Get folds
kf = KFold(n_splits=10)
folds = kf.split(data)

##############
# Create output structure
main_out_dir = os.path.join(args.output_dir, 'sk', args.label_preproc)

#############
# Train models and report
fold = 0
for i_train, i_test in folds:
    X_train = X[i_train]
    Y_train = np.array(Y[i_train], dtype=float)
    X_test = X[i_test]
    Y_test = np.array(Y[i_test], dtype=float)

    # Scale Y if asked for
    if args.label_preproc == "scale":
        Y_scaler = pp.StandardScaler()
        Y_scaler.fit(Y_train)
        Y_train = Y_scaler.transform(Y_train)

    # Train model
    k = flakes.wrappers.gpy.GPyStringKernel(gap_decay=0.1, match_decay=0.1, order_coefs=[1.0] * 5, 
                                            embs=embs, device='/cpu:0', mode='tf-batch', 
                                            batch_size=10, sim='dot', 
                                            wrapper='none')
    if args.label_preproc == "warp":
        model = GPy.models.WarpedGP(X_train, Y_train, kernel=k)
        model['warp_tanh.psi'] = np.random.lognormal(0, 1, (3, 3))
    else:
        model = GPy.models.GPRegression(X_train, Y_train, kernel=k)
    model.optimize(messages=True, max_iters=100)

    # Get predictions
    info_dict = {}
    preds, vars = model.predict_noiseless(X_test)
    if args.label_preproc == 'scale':
        preds = Y_scaler.inverse_transform(preds)
    info_dict['mae'] = MAE(preds, Y_test)
    info_dict['rmse'] = np.sqrt(MSE(preds, Y_test))
    info_dict['pearsonr'] = pearsonr(preds.flatten(), Y_test.flatten())
    lpd = model.log_predictive_density(X_test, Y_test)
    info_dict['nlpd'] = -np.mean(lpd)

    # Get parameters
    print model
    info_dict['gap_decay'] = float(model['string.gap_decay'])
    info_dict['match_decay'] = float(model['string.match_decay'])
    info_dict['coefs'] = list(model['string.coefs'])
    info_dict['noise'] = float(model['Gaussian_noise.variance'])
    info_dict['log_likelihood'] = float(model.log_likelihood())
    if args.label_preproc == 'warp':
        info_dict['warp_psi'] = list([list(pars) for pars in model['warp_tanh.psi']])
        info_dict['warp_d'] = float(model['warp_tanh.d'])

    # Save information
    fold_dir = os.path.join(main_out_dir, str(fold))
    try:
        os.makedirs(fold_dir)
    except OSError:
        # Already exists
        pass
    with open(os.path.join(fold_dir, 'info.json'), 'w') as f:
        json.dump(info_dict, f, indent=2)
    np.savetxt(os.path.join(fold_dir, 'preds.tsv'), preds)
    np.savetxt(os.path.join(fold_dir, 'vars.tsv'), vars)

    # Cleanup
    model.kern._implementation.sess.close()
    del model

    # Next fold
    fold += 1
