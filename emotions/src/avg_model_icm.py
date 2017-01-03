"""
Models that employ an average of word embeddings as inputs.
This employs ICM models over the 6 emotions.

"""
import numpy as np
import sys
import os
import argparse
import json
import gc

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
parser.add_argument('--rank', help='rank of coreg matrix', default=1, type=int)
parser.add_argument('--data_size', help='size of dataset, default is full size',
                    default=10000, type=int)
parser.add_argument('--ard', help='set this flag to enable ARD', action='store_true')
parser.add_argument('--bias', help='set this flag to add a bias kernel', action='store_true')
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
Y = data[:, -6:]
#print Y
##############
# Get folds
kf = KFold(n_splits=10)
folds = kf.split(data)

##############
# Create output structure
#if args.ard:
#    mode = 'ard'
#else:
#    mode = 'iso'
mode = 'iso'
if args.bias:
    mode += '_bias'
main_out_dir = os.path.join(args.output_dir, 'avg', 'icm', args.model + '_' + mode, args.label_preproc)

#############
# Train models and report
fold = 0
for i_train, i_test in folds:
    X_train = X[i_train]
    Y_train = Y[i_train]
    X_test = X[i_test]
    Y_test = Y[i_test]

    # Preprocess data as input lists
    X_train_list = []
    Y_train_list = []
    X_test_list = []
    Y_test_list = []

    for emo_id, emo in enumerate(EMOS):
        X_train_list.append(np.copy(X_train))
        Y_train_list.append(Y_train[:, emo_id:emo_id+1])
        X_test_list.append(np.copy(X_test))
        Y_test_list.append(Y_test[:, emo_id:emo_id+1])

    # Scale Y if asked for
    # if args.label_preproc == "scale":
    #     Y_scaler_list = []
    #     for emo_id, emo in enumerate(EMOS):
    #         Y_scaler = pp.StandardScaler()
    #         Y_scaler.fit(Y_train_list[emo_id])
    #         Y_train_list[emo_id] = Y_scaler.transform(Y_train_list[emo_id])
    #         Y_scaler_list.append(Y_scaler)
    # elif args.label_preproc == "warp":
    #     # Warped GPs seems to break if we have too many zeroes.
    #     Y_train_list = [Y_train - 50 for Y_train in Y_train_list]
    
    # Select and train model
    # TODO: implement ridge and svr using EasyAdapt
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
        #if args.label_preproc == "warp":
        #    model = GPy.models.WarpedGP(X_train, Y_train, kernel=k)
        #    model['warp_tanh.psi'] = np.random.lognormal(0, 1, (3, 3))
        #else:
            #model = GPy.models.GPRegression(X_train, Y_train, kernel=k)
        icmk = GPy.util.multioutput.ICM(input_dim=X.shape[1], num_outputs=6, 
                                        kernel=k, W_rank=args.rank)
        model = GPy.models.GPCoregionalizedRegression(X_train_list,
                                                      Y_train_list,
                                                      kernel=icmk)
        model.optimize(messages=True, max_iters=100)
        print model
    
        # Get predictions
        info_dict = {}
        preds_list = []
        vars_list = []
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
            #X_test_pred, Y_test_pred, index = GPy.util.multioutput.build_XY(X_test_list, Y_test_list)
            #noise_dict = {'output_index': X_test_pred[:,-1:].astype(int)}
            #preds, vars = model.predict_noiseless(X_test, Y_metadata=noise_dict)
            for emo_id, emo in enumerate(EMOS):
                # TODO: preprocessing
                emo_dict = {}
                to_predict = np.concatenate((X_test_list[emo_id], np.ones((X_test.shape[0], 1)) * emo_id), axis=1)
                noise_dict = {'output_index': np.ones((X_test.shape[0], 1), dtype=int) * (emo_id)}
                preds, vars = model.predict(to_predict, Y_metadata=noise_dict)
                #if args.label_preproc == 'scale':
                #    preds = Y_scaler_list[emo_id].inverse_transform(preds)
                emo_dict['mae'] = MAE(preds, Y_test_list[emo_id])
                emo_dict['rmse'] = np.sqrt(MSE(preds, Y_test_list[emo_id]))
                emo_dict['pearsonr'] = pearsonr(preds.flatten(), Y_test_list[emo_id].flatten())
                #Y_metadata = {}
                #Y_metadata['output_index'] = np.ones(X_test.shape[0]) * emo_id
                emo_dict['nlpd'] = -np.mean(model.log_predictive_density(to_predict, 
                                                                         Y_test_list[emo_id],
                                                                         Y_metadata=noise_dict))
                info_dict[emo] = emo_dict
                preds_list.append(preds.flatten())
                vars_list.append(vars.flatten())

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
                if p_name == 'ICM.B.W':
                    info_dict[p_name] = list([list(pars) for pars in model[p_name]])
                else:
                    try:
                        info_dict[p_name] = float(model[p_name])
                    except TypeError: #ARD
                        info_dict[p_name] = list(model[p_name])
            info_dict['log_likelihood'] = float(model.log_likelihood())
        # elif args.model == 'rbf':
        #     info_dict['variance'] = float(model['ICM.rbf.variance'])
        #     info_dict['lengthscale'] = list(model['ICM.rbf.lengthscale'])
        #     info_dict['noise'] = list([float(noise) for noise in model['mixed_noise.*']])
        #     info_dict['log_likelihood'] = float(model.log_likelihood())
        # elif args.model == 'mat32':
        #     info_dict['variance'] = float(model['ICM.Mat32.variance'])
        #     info_dict['lengthscale'] = list(model['ICM.Mat32.lengthscale'])
        #     info_dict['noise'] = list([float(noise) for noise in model['mixed_noise.*']])
        #     info_dict['log_likelihood'] = float(model.log_likelihood())
        # elif args.model == 'mat52':
        #     info_dict['variance'] = float(model['ICM.Mat52.variance'])
        #     info_dict['lengthscale'] = list(model['ICM.Mat52.lengthscale'])
        #     info_dict['noise'] = list([float(noise) for noise in model['mixed_noise.*']])
        #     info_dict['log_likelihood'] = float(model.log_likelihood())
        # elif args.model == 'ratquad':
        #     info_dict['variance'] = float(model['ICM.RatQuad.variance'])
        #     info_dict['lengthscale'] = list(model['ICM.RatQuad.lengthscale'])
        #     info_dict['power'] = list(model['ICM.RatQuad.power'])
        #     info_dict['noise'] = list([float(noise) for noise in model['mixed_noise.*']])
        #     info_dict['log_likelihood'] = float(model.log_likelihood())
        # elif args.model == 'linear':
        #     info_dict['variance'] = list(model['ICM.linear.variances'])
        #     info_dict['noise'] = list([float(noise) for noise in model['mixed_noise.*']])
        #     info_dict['log_likelihood'] = float(model.log_likelihood())
        # elif args.model == 'mlp':
        #     info_dict['variance'] = float(model['ICM.mlp.variance'])
        #     info_dict['weight_variance'] = list(model['ICM.mlp.weight_variance'])
        #     info_dict['bias_variance'] = float(model['ICM.mlp.bias_variance'])
        #     info_dict['noise'] = list([float(noise) for noise in model['mixed_noise.*']])
        #     info_dict['log_likelihood'] = float(model.log_likelihood())

        #if args.model != 'ridge' and args.model != 'svr':
        #    info_dict['W'] = list([list(w) for w in model['ICM.B.W']])
        #    info_dict['kappa'] = list([float(kappa) for kappa in model['ICM.B.kappa']])

        #if args.label_preproc == 'warp':
        #    info_dict['warp_psi'] = list([list(pars) for pars in model['warp_tanh.psi']])
        #    info_dict['warp_d'] = float(model['warp_tanh.d'])
        
    
        # Save information
        fold_dir = os.path.join(main_out_dir, 'rank_' + str(args.rank), str(fold))
        try:
            os.makedirs(fold_dir)
        except OSError:
            # Already exists
            pass
        with open(os.path.join(fold_dir, 'info.json'), 'w') as f:
            json.dump(info_dict, f, indent=2)
        preds_list = np.array(preds_list)
        np.savetxt(os.path.join(fold_dir, 'preds.tsv'), np.transpose(preds_list))
        if args.model != 'ridge' and args.model != 'svr':
            np.savetxt(os.path.join(fold_dir, 'vars.tsv'), np.transpose(vars_list))

        gc.collect(2)
            
    # Finished emotions, next fold
    fold += 1
