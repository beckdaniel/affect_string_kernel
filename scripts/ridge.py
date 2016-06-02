from sklearn.linear_model import Ridge, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from scipy.stats.stats import pearsonr
import sys
import util
import numpy as np

INPUTS = sys.argv[1]
LABELS = sys.argv[2]
EMBS = sys.argv[3]
SPLIT = 900

###################
# LOAD EVERYTHING

embs = util.load_embs(EMBS)
X = []

with open(INPUTS) as f:
    for line in f:
        X.append(util.preprocess_sent(line.split('_')[1]))

Y = np.loadtxt(LABELS)[:, 1]

###################
# PREPROCESS X
X  = np.array([util.average_sent(sent, embs) for sent in X])
#print X
#print X.shape

####################
# RIDGE
m = RidgeCV()
#m = KernelRidge(kernel='rbf')
#m = Ridge()
m.fit(X[:SPLIT], Y[:SPLIT])
preds = m.predict(X[SPLIT:])
Y_test = Y[SPLIT:]

for tup in zip(preds, Y_test)[:20]:
    print tup
print MAE(preds, Y_test)
print np.sqrt(MSE(preds, Y_test))
print pearsonr(preds, Y_test)

