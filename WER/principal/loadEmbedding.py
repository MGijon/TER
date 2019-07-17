# Python basic package utilities
import os
import pickle as plk
import random
import pandas as pd
from tempfile import TemporaryFile
import heapq
import logging
import functools
import time
# NLTK package
from nltk.corpus import wordnet as wn
# Scipy package
import scipy.spatial.distance
from scipy.stats import entropy
from scipy import stats
# Numpy package
import numpy as np
from numpy.linalg import norm
# Gensim package
import gensim
import gensim.models as gm
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
# Sklearn package
import sklearn.metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import normalize

# Scipy package
#from scipy import stats


def __init__(self, path='Embeddings/', embedings_size=300, type='GloVe', log=''):


    allowed_types = {'GloVe':1,
                     'Word2Vec':2}

    self.path = path
    self.embedings_size = embedings_size
    self.type = allowed_types[type]
    self.embeddings_index = {}
    self.words = []
    self.filtered_words = []
    self.synonims = []
    self.synonimsDistribution = []
    self.random_words_pairs = []
    self.randomDistribution = []
    self.antonims = []
    self.antonimsDistribution = []
    self.wordSynset = []

    # looger
    # ------
    logger_name = log

    self.logger = logging.getLogger(logger_name)
    self.logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logger_name + '.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    self.logger.addHandler(fh)

    # GLOVE
    # =====
    if self.type == 1:

        self.embedding_name = 'glove.6B.' + str(self.embedings_size) + 'd.txt'
        f = open(os.path.join(self.path, self.embedding_name))

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs

        f.close()
        self.words = list(self.embeddings_index.keys())

        self.logger.info('-. GloVe embedding .-\n')

    # WORD2VEC
    # ========
    elif self.type == 2:
        self.model = gensim.models.KeyedVectors.load_word2vec_format(path + 'GoogleNews-vectors-negative300.bin.gz', binary=True)
        self.words = list(self.model.vocab)

        self.logger.info('-. Word2Vec embedding .-\n')

    else:
        print('ERROR')
        self.logger.info('FATAL ERROR, the embedding has not been charged for some unknown reason.')
