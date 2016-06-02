import numpy as np
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import _treebank_word_tokenize
import scipy as sp
import sys
import os


def preprocess_sent(sent):#, lemmatizer):
    """
    Take a sentence in string format and returns
    a list of lemmatized tokens.
    """
    tokenized = _treebank_word_tokenize(sent.lower())
    #sent = [lemmatizer.lemmatize(word) for word in tokenized]
    return tokenized


def read_embeddings(embs_file):
    embs = {}
    with open(embs_file) as f:
        for line in f:
            tokens = line.split()
            embs[tokens[0]] = [float(val) for val in tokens[1:]]
    return embs


def build_sent_matrix(word_list):
    matrix = []
    for word in word_list:
        if word in EMBS:
            matrix.append(EMBS[word])
        else:
            matrix.append([0.0] * 50)
    return np.array(matrix)



DATA_FILE = sys.argv[1]
EMBS_FILE = sys.argv[2]
PREPROC_DIR = sys.argv[3]
EMBS = read_embeddings(EMBS_FILE)

#preproc_data = []
with open(DATA_FILE) as f:
    for line in f:
        sid, sent = line.strip().split('_')
        emb_sent = build_sent_matrix(preprocess_sent(sent))
        np.savetxt(os.path.join(PREPROC_DIR, sid + '.emb'), emb_sent)
        #preproc_data.append(emb_sent)
        
