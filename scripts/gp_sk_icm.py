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
SPLIT = 10
TEST_SPLIT = 10

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
X_test = X[-TEST_SPLIT:]
Y_test = Y[-TEST_SPLIT:]

print X_train.ndim

######################
X_train_list = []
Y_train_list = []
X_test_list = []
Y_test_list = []
EMOS = ['sadness','fear','anger','disgust','surprise','joy']

for emo_id, emo in enumerate(EMOS):
    #for emo in sorted(EMOS): # very important to sort here
        #emo_id = EMO_DICT[emo]
    X_train_list.append(np.copy(X_train))
    Y_train_list.append(Y_train[:, emo_id:emo_id+1])
    X_test_list.append(np.copy(X_test))
    Y_test_list.append(Y_test[:, emo_id:emo_id+1])


####################
sk = flakes.wrappers.gpy.GPyStringKernel(order_coefs=[1.0] * 4, embs=embs, mode='tf-batch')
k = GPy.util.multioutput.ICM(input_dim=X_train.shape[1], num_outputs=6, 
                             kernel=sk, W_rank=1)

m = GPy.models.GPCoregionalizedRegression(X_list=X_train_list, 
                                          Y_list=Y_train_list,
                                          kernel=k)

print m
#m.optimize_restarts(num_restarts=5, robust=True, messages=True, max_iters=30)
#m.optimize(messages=True, max_iters=100)
print m
print m['.*coefs.*']

X_test, Y_test, index = GPy.util.multioutput.build_XY(X_test_list, Y_test_list)
noise_dict = {'output_index': X_test[:,1:].astype(int)}

preds = m.predict(X_test, Y_metadata=noise_dict)[0].flatten()
Y_test = np.array(Y_test).flatten()
print len(preds)
for tup in zip(preds, Y_test):
    print tup
print MAE(preds, Y_test)
print np.sqrt(MSE(preds, Y_test))
print pearsonr(preds, Y_test)

