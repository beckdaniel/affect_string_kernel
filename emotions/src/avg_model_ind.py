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



############
# Parse args

parser = argparse.ArgumentParser()
parser.add_argument('inputs', help='the file with the sentence inputs')
parser.add_argument('labels', help='the file with the valence labels')
parser.add_argument('embs', help= 'the word embeddings file')
parser.add_argument('model', help='one of: "ridge", "svr", "linear",' +
                    '"rbf", "mat32", "mat52", "ratquad", "mlp"')
parser.add_argument('label_preproc', help='one of "none", "scale", "warp"')
parser.add_argument('output_dir', help='directory where outputs will be stored')
parser.add_argument('--data_size', help='size of dataset, default is full size',
                    default=10000, type=int)
parser.add_argument('--ard', help='set this flag to enable ARD', action='store_true')
parser.add_argument('--bias', help='set this flag to add a bias kernel', action='store_true')
parser.add_argument('--norm', help='set this flag to normalise the labels', action='store_true')
args = parser.parse_args()

###########
# Constants

# This is the emotion label order in the original data.
EMOS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

###########
# Load data

embs = util.load_embs(args.embs)
X = []
with open(args.inputs) as f:
    for line in f:
        X.append(util.preprocess_sent(line.split('_')[1]))
Y = np.loadtxt(args.labels)[:, 1:]
#print Y

###############
# Preprocessing
X = np.array([util.average_sent(sent, embs) for sent in X])
data = np.concatenate((X, Y), axis=1)[:args.data_size]
np.random.seed(1000)
np.random.shuffle(data)
X = data[:, :-6]
Y = (data[:, -6:])
if args.norm:
    Y = (Y + 1)
#print Y
##############
# Get folds
kf = KFold(n_splits=10)
folds = kf.split(data)

##############
# Create output structure
if args.ard:
    mode = 'ard50'
else:
    mode = 'iso50'
if args.bias:
    mode += '_bias'
if args.norm:
    mode += '_norm'
main_out_dir = os.path.join(args.output_dir, 'avg', 'ind', args.model + '_' + mode, args.label_preproc)

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

        # Scale Y if asked for
        if args.label_preproc == "scale":
            Y_scaler = pp.StandardScaler()
            Y_scaler.fit(Y_train)
            Y_train = Y_scaler.transform(Y_train)
        #elif args.label_preproc == "warp":
            # Warped GPs seems to break if we have too many zeroes.
        #    Y_train -= 50
    
        # Select and train model
        if args.model == 'ridge':
            model = RidgeCV(alphas=np.logspace(-2, 2, 5))
            #print X_train
            #print Y_train
            model.fit(X_train, Y_train.flatten())
        elif args.model == 'svr':
            hypers = {'C': np.logspace(-2, 2, 5),
                      'epsilon': np.logspace(-3, 1, 5),
                      'gamma': np.logspace(-3, 1, 5)}
            model = GridSearchCV(SVR(), hypers)
            model.fit(X_train, Y_train.flatten())
        else:
            if args.model == 'rbf':
                k = GPy.kern.RBF(X.shape[1], ARD=args.ard)
            elif args.model == 'mat32':
                k = GPy.kern.Matern32(X.shape[1], ARD=args.ard)
            elif args.model == 'mat52':
                k = GPy.kern.Matern52(X.shape[1], ARD=args.ard)
            elif args.model == 'ratquad':
                k = GPy.kern.RatQuad(X.shape[1], ARD=args.ard)
            elif args.model == 'linear':
                k = GPy.kern.Linear(X.shape[1], ARD=args.ard)
            elif args.model == 'mlp':
                k = GPy.kern.MLP(X.shape[1], ARD=args.ard)
                
            if args.bias:
                k = k + GPy.kern.Bias(X.shape[1])
                
            if args.label_preproc == "warp":
                model = GPy.models.WarpedGP(X_train, Y_train, kernel=k)
                warp_f = GPy.util.warping_functions.TanhFunction(n_terms=1)
                model['warp_tanh.psi'] = np.random.lognormal(0, 1, (1, 1))
            else:
                model = GPy.models.GPRegression(X_train, Y_train, kernel=k)
            model.optimize(messages=True, max_iters=100)
    
        # Get predictions
        info_dict = {}
        if args.model == 'ridge' or args.model == 'svr':
            preds = model.predict(X_test)
            if args.label_preproc == 'scale':
                preds = Y_scaler.inverse_transform(preds)
            elif args.label_preproc == 'warp':
                preds += 50
            info_dict['mae'] = MAE(preds, Y_test.flatten())
            info_dict['rmse'] = np.sqrt(MSE(preds, Y_test.flatten()))
            info_dict['pearsonr'] = pearsonr(preds, Y_test.flatten())
        else:
            # TODO: check if this makes sense
            #preds, vars = model.predict(X_test)
            if args.label_preproc == 'warp':
                preds, vars = model.predict(X_test, median=True)
            else:
                #preds, vars = model.predict_noiseless(X_test)
                preds, vars = model.predict(X_test)
                print preds
                print vars
            #if args.label_preproc == 'scale':
            #    preds = Y_scaler.inverse_transform(preds)
            #elif args.label_preproc == 'warp':
            #    Y_test -= 50
            info_dict['mae'] = MAE(preds, Y_test)
            info_dict['rmse'] = np.sqrt(MSE(preds, Y_test))
            info_dict['pearsonr'] = pearsonr(preds.flatten(), Y_test.flatten())
            nlpd = -model.log_predictive_density(X_test, Y_test)
            info_dict['nlpd'] = np.mean(nlpd)

        # Get parameters
        if args.model == 'ridge':
            info_dict['coefs'] = list(model.coef_)
            info_dict['intercept'] = model.intercept_
            info_dict['regularization'] = model.alpha_
        elif args.model == 'svr':
            info_dict['regularization'] = model.best_params_['C']
            info_dict['epsilon'] = model.best_params_['epsilon']
            info_dict['gamma'] = model.best_params_['gamma']
        else:
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
        # elif args.model == 'rbf':
        #     info_dict['variance'] = float(model['rbf.variance'])
        #     info_dict['lengthscale'] = list(model['rbf.lengthscale'])
        #     info_dict['noise'] = float(model['Gaussian_noise.variance'])
        #     info_dict['log_likelihood'] = float(model.log_likelihood())
        # elif args.model == 'mat32':
        #     info_dict['variance'] = float(model['Mat32.variance'])
        #     info_dict['lengthscale'] = list(model['Mat32.lengthscale'])
        #     info_dict['noise'] = float(model['Gaussian_noise.variance'])
        #     info_dict['log_likelihood'] = float(model.log_likelihood())
        # elif args.model == 'mat52':
        #     info_dict['variance'] = float(model['Mat52.variance'])
        #     info_dict['lengthscale'] = list(model['Mat52.lengthscale'])
        #     info_dict['noise'] = float(model['Gaussian_noise.variance'])
        #     info_dict['log_likelihood'] = float(model.log_likelihood())
        # elif args.model == 'ratquad':
        #     info_dict['variance'] = float(model['RatQuad.variance'])
        #     info_dict['lengthscale'] = list(model['RatQuad.lengthscale'])
        #     info_dict['power'] = list(model['RatQuad.power'])
        #     info_dict['noise'] = float(model['Gaussian_noise.variance'])
        #     info_dict['log_likelihood'] = float(model.log_likelihood())
        # elif args.model == 'linear':
        #     info_dict['variance'] = list(model['linear.variances'])
        #     info_dict['noise'] = float(model['Gaussian_noise.variance'])
        #     info_dict['log_likelihood'] = float(model.log_likelihood())
        # elif args.model == 'mlp':
        #     info_dict['variance'] = float(model['mlp.variance'])
        #     info_dict['weight_variance'] = list(model['mlp.weight_variance'])
        #     info_dict['bias_variance'] = float(model['mlp.bias_variance'])
        #     info_dict['noise'] = float(model['Gaussian_noise.variance'])
        #     info_dict['log_likelihood'] = float(model.log_likelihood())


        #if args.label_preproc == 'warp':
        #    info_dict['warp_psi'] = list([list(pars) for pars in model['warp_tanh.psi']])
        #    info_dict['warp_d'] = float(model['warp_tanh.d'])
        
    
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
        if args.model != 'ridge' and args.model != 'svr':
            np.savetxt(os.path.join(fold_dir, 'vars.tsv'), vars)

    # Finished emotions, next fold
    fold += 1
