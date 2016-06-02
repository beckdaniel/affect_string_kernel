import numpy as np
import flakes
import GPy
import sys
import os

def read_sents(sents_dir):
    X = []
    for i in range(500, 1001):
        sent_file = os.path.join(sents_dir, str(i) + '.emb')
        X.append([np.loadtxt(sent_file)])
    #print X[:1]
    return np.array(X)

def pad_sents(X):
    maxlen = max([len(x[0]) for x in X])
    new_X = []
    for x in X:
        pads = maxlen - len(x[0])
        #print pads
        #print maxlen
        new_x = np.concatenate((x[0], np.zeros(shape=(pads, 50))), axis=0)
        new_X.append([new_x])
    return np.array(new_X)

SENTS_DIR = sys.argv[1]
LABELS_FILE = sys.argv[2]

INSTANCES = 200
X = read_sents(SENTS_DIR)[:INSTANCES]
X = pad_sents(X)
Y = np.loadtxt(LABELS_FILE)[:INSTANCES, -1:]

print "FINISH READING"

sk = flakes.wrappers.gpy.GPyStringKernel()
sk.order_coefs = [1.0, 0.5]
sk.decay = 0.1

bias = GPy.kern.Bias(1)

m = GPy.models.GPRegression(X, Y, kernel=sk+bias)
m.optimize(messages=True, max_iters=20)
result = m.predict(X)
print result
print Y
print m
