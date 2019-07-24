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


def loadEmbedding(path='Embeddings/', embeding_name='', embedings_size, type='GloVe'):

    # debería hacer más robusto el tema del nombre (aceptar cosas como GloVE)
    # ya me aseguraré en el futuro de   que haga más cosas
    type = type.lower()
    allowed_types = {'glove':1,
                     'word2vec':2}

    embedding_dictionary = {} # element to return

    try:
        # GloVe
        # -----
        if type == 1 or allowed_types[type] == 1:
            print('el tipo de embedding lo pilla sin problemas')
            embedding_dictionary['dimension'] = embedding_size
            embedding_dictionary['embedding_name'] = embedding_name
            print(embedding_dictionary['embedding_name'])
            # loading the embedding
            try:
                f = open(os.path.join(path, embedding_dictionary['embedding_name']))

                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embedding_dictionary['embeddings_index'][word] = coefs
                    f.close()
            except FileNotFoundError as fnf_error:
                print(fnf_error)
            finally:
                print('borrar esta parte pronto, no ccreo que me haga dalta')
            embedding_dictionary['words'] = list(embeddings_index.keys())


        # Word2Vec
        # --------
        elif type == 2 or allowed_types[type] == 2:
            print('el tipo de embedding lo pilla sin problemas')
            embedding_dictionary['dimension'] = embedding_size
            print(embedding_dictionary['dimension'])
            embedding_dictionary['model'] = gensim.models.KeyedVectors.load_word2vec_format(
                                            path + 'GoogleNews-vectors-negative300.bin.gz',
                                            binary=True)
            embedding_dictionary['words'] = list(embedding_dictionary['model'].vocab)

        else:
            print('Fatal error loading the embedding')


        return embedding_dictionary

    except:
        print('Error loading the embedding')
