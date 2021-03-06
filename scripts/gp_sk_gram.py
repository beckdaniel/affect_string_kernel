from sklearn.linear_model import Ridge, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from scipy.stats.stats import pearsonr
import sys
import util
import numpy as np
import GPy
import flakes

INPUTS = sys.argv[1]
LABELS = sys.argv[2]
EMBS = sys.argv[3]
SPLIT = 100
TEST_SPLIT = 100

###################
# LOAD EVERYTHING

embs = util.load_embs(EMBS)
X = []

with open(INPUTS) as f:
    for line in f:
        X.append([util.preprocess_sent(line.split('_')[1])])

Y = np.loadtxt(LABELS)[:, 1:]

###################
# PREPROCESS X
#X  = np.array([util.average_sent(sent, embs) for sent in X])
X = np.array(X, dtype=object)
X_train = X[:SPLIT]
Y_train = Y[:SPLIT]
X_test = X[SPLIT:]
Y_test = Y[SPLIT:].flatten()

print X_train.ndim

####################
# RIDGE
#sk = flakes.wrappers.gpy.GPyStringKernel(order_coefs=[1.0] * 5, embs=embs, device='/gpu:7', mode='tf-gram')
#alphabet = set([word for sent in X_train for word in sent[0]])
sk = flakes.wrappers.gpy.GPyStringKernel(order_coefs=[1.0] * 6, embs=embs, mode='tf-batch')
m = GPy.models.GPRegression(X_train, Y_train, kernel=sk)
print m
#m.optimize_restarts(num_restarts=5, robust=True, messages=True, max_iters=30)
m.optimize(messages=True, max_iters=30)
print m
print m['.*coefs.*']
preds = m.predict(X_test)[0].flatten()

for tup in zip(preds, Y_test):
    print tup
print MAE(preds, Y_test)
print np.sqrt(MSE(preds, Y_test))
print pearsonr(preds, Y_test)

