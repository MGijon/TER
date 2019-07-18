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

def returnVector(embedding_dictionay , type, setOfWords):

    vectorsArray = []

    try:
        # GloVe
        if type == 1:
            for word in setOfWords:
                vectorsArray.append(embeddings_index[word])   # arreglar esto

        # Word2Vec
        # ========
        elif type == 2:
            for word in setOfWords:
                vectorsArray.append(self.model.get_vector(word))   # arreglar esto
    except:
        message = 'Sorry, in this list of words there is at least one word that is not in the vocabulary of the embedding'
        print(message)
        self.logger.info('Failed returnVector function - some word not in vocabulary\n')
        pass # to let the programm continue

    return vectorsArray
