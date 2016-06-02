import numpy as np
from nltk.tokenize import _treebank_word_tokenize


def load_embs(filename):
    embs = {}
    with open(filename) as f:
        for line in f:
            toks = line.split()
            embs[toks[0]] = np.array([float(t) for t in toks[1:]])
    return embs


def preprocess_sent(sent):
    """
    Take a sentence in string format and returns
    a list of lemmatized tokens.
    """
    tokenized = _treebank_word_tokenize(sent.lower())
    return tokenized


def average_sent(sent, embs):
    result = []
    for word in sent:
        try:
            result.append(embs[word])
        except KeyError:
            pass
    return np.mean(result, axis=0)
    
    
