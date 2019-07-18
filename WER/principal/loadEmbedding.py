""" Devolverá un objeto tipo diccionario con muchas cosas necesarias todas ellas """

# Python basic package utilities
import os
import pickle as plk
import random
import pandas as pd
from tempfile import TemporaryFile
import heapq
#import logging
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


def loadEmbedding(path='Embeddings/', embedding_name='', embedings_size=300, type='GloVe'):

    # debería hacer más robusto el tema del nombre (aceptar cosas como GloVE)
    # ya me aseguraré en el futuro de   que haga más cosas
    allowed_types = {'GloVe':1, 'Word2Vec':2}

    embedding_dictionary = {} # element to return

    # GloVe
    # -----
    if type == 1:

        embedding_dictionary['embedding_name'] = 'glove.6B.' + str(self.embedings_size) + 'd.txt'
        f = open(os.path.join(path, embedding_dictionary['embedding_name']))

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_dictionary['embeddings_index'][word] = coefs

        f.close()
        embedding_dictionary['words'] = list(embeddings_index.keys())


    # Word2Vec
    # --------
    elif type == 2:
        embedding_dictionary['model'] = gensim.models.KeyedVectors.load_word2vec_format(
                                        path + 'GoogleNews-vectors-negative300.bin.gz',
                                        binary=True)
        embedding_dictionary['words'] = list(embedding_dictionary['model'].vocab)

    else:
        print('Fatal error loading the embedding')


    return embedding_dictionary
