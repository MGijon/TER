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

def loadEmbedding(embeding_name='', embedings_size=300, path='Embeddings/',  type='GloVe'):

    # debería hacer más robusto el tema del nombre (aceptar cosas como GloVE)
    # ya me aseguraré en el futuro de   que haga más cosas
    type = type.lower()
    allowed_types = {'glove':1,
                     'word2vec':2}

    dict = {
        'dimension':0,
        'embeding_name': '',
        'model': [],
        'words': [],
        'embeddings_index': [], # solo en el caso de GloVe
    } # element to return

    try:
        # GloVe : NO ACABA DE FUNCIONAR CORRECTAMENTE
        # -----
        if type == 1 or allowed_types[type] == 1:
            dict['dimension'] = embedings_size
            dict['embeding_name'] = embeding_name
            #print(dict['embeding_name'])
            #print(path)
            # loading the embedding
            with open(os.path.join(path, embeding_name)) as f:
            #print(f)
            #print(os.path.join(path, dict['embeding_name']))
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    dict['embeddings_index'] = coefs
                f.close()


            dict['words'] = list(dict['embeddings_index'].keys())


        # Word2Vec
        # --------
        elif type == 2 or allowed_types[type] == 2:
            dict['dimension'] = embedings_size
            dict['embeding_name'] = embeding_name
            print(path + dict['embeding_name'])
            dict['model'] = gensim.models.KeyedVectors.load_word2vec_format(
                                            path + embeding_name,
                                            binary=True)
            dict['words'] = list(dict['model'].vocab)

        else:
            print('Fatal error loading the embedding')


        return dict

    except Exception as e:
        print(e)
