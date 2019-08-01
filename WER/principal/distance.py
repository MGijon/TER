''' Compute the distance between two vector1s under the selected norm.
:param vector1: (array, floats) self-explanatory.
:param vector2: (array, floats) self-explanatory.
:param norm: distance
:return: value of the distence (under the selected norm) between the two vector1s
'''
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
#from gensim.models import Keyedvectors
from gensim.scripts.glove2word2vec import glove2word2vec
# Sklearn package
import sklearn.metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import normalize

def distance(vector1, vector2, norm=1):

    if norm == 1 or norm is "euclidean":

        calculo = vector1
        for i in range(0, len(vector1)):
            calculo[i] = calculo[i] - vector2[i]
        suma = 0
        for i in calculo:
            suma += np.power(i, 2)
        value = np.sqrt(suma)

    elif norm == 2 or norm is "cosine":
        value = scipy.spatial.distance.cosine(vector1, vector2)

    elif norm == 3 or norm is "cityblock":
        value = scipy.spatial.distance.cityblock(vector1, vector2)

    elif norm == 4 or norm is "l1":
         value = np.linalg.norm((vector1 - vector2), ord=1)

    elif norm == 7 or norm is "chebyshev":
        value = scipy.spatial.distance.chebyshev(vector1, vector2)

    elif norm == 8 or norm is "minkowski":
        value = scipy.spatial.distance.minkowski(vector1, vector2)

    elif norm == 9 or norm is "sqeuclidean":
        value = scipy.spatial.distance.sqeuclidean(vector1, vector2)

    elif norm == 10 or norm is "jensenshannon":
        _P = vector1 / norm(vector1, ord=1)
        _Q = vector2 / norm(vector2, ord=1)
        _M = 0.5 * (_P + _Q)
        value =  0.5 * (entropy(_P, _M) + entropy(_Q, _M))

    elif norm == 12 or norm is "jaccard":
        sklearn.metrics.jaccard_similarity_score(vector1, vector2)

    elif norm == 13 or norm is "correlation":
        value = scipy.spatial.distance.correlation(vector1, vector2)

    elif norm == 14 or norm is "braycurtis":
        value = scipy.spatial.distance.braycurtis(vector1, vector2)

    elif norm == 15 or norm is "canberra":
        value = scipy.spatial.distance.canberra(vector1, vector2)

    elif norm == 16 or norm is "kulsinski":
        value = scipy.spatial.distance.cdis(vector1, vector2)

    elif norm == 17 or norm is "max5":
        # take the sum of the 5 maximun difference dimensions
        v = vector2 - vector1
        v2 = [abs(x) for x in v]
        aux = heapq.nlargest(5, v2)
        value = sum(aux)

    elif norm == 18 or norm is "max10":
        # take the sum of the 10 maximun difference dimensions
        v = vector2 - vector1
        v2 = [abs(x) for x in v]
        aux = heapq.nlargest(10, v2)
        value = sum(aux)


    elif norm == 19 or norm is "max25":
        # take the sum of the 25 maximun difference dimensions
        v = vector2 - vector1
        v2 = [abs(x) for x in v]
        aux = heapq.nlargest(25, v2)
        value = sum(aux)

    elif norm == 20 or norm is "max50":
        # take the sum of the 50 maximun difference dimensions
        v = vector2 - vector1
        v2 = [abs(x) for x in v]
        aux = heapq.nlargest(50, v2)
        value = sum(aux)

    elif norm == 21 or norm is "max100":
        # take the sum of the 100 maximun difference dimensions
        v = vector2 - vector1
        v2 = [abs(x) for x in v]
        aux = heapq.nlargest(100, v2)
        value = sum(aux)

    elif norm == 28:
        non_sing_changes = 0

        for i in range(0, len(vector1)):
            if vector1[i] >= 0 and vector2[i] >= 0:
                non_sing_changes += 1
            if vector1[i] < 0 and vector2[i] < 0:
                non_sing_changes += 1

        value = len(vector1) - non_sing_changes

    elif norm == 29:
        epsilon = 0
        for coordinate in range(0, len(vector1)):
            auxiliar = abs(vector1[coordinate] - vector2[coordinate])
            if auxiliar > epsilon:
                epsilon = auxiliar
        value = epsilon

    elif norm == 30:
        epsions = 0
        for coordinate in range(0, len(vector1)):
            epsions += abs(vector1[coordinate] - vector2[coordinate])
        value = epsions / len(vector1)


    elif norma == 31:
        differencevector1 = [abs(vector1[i] - vector2[i]) for i in range(0, len(vector1))]
        value = differencevector1

    else:
        pass

    return value
