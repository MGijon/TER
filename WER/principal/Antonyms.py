""" """

# BORRARÉ LUEGO LAS DEPENDENCIAS INNECESARIAS !!!!


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


def pureAntonyms(self, words):
    '''
    Just compute the set of synonims, without distances
    :return: None
    '''

    conjunto = []
    for target_word in words:
        synsets = wn.synsets(target_word)
        for synset in synsets:
            auxiliar = []
            lemmas = synset.lemmas()
            numberSynom = len(lemmas)
            if numberSynom > 1:
                for lemma in lemmas:
                    if lemma.antonyms():
                        antonimo = lemma.antonyms()[0].name()
                        if antonimo in words:
                            auxiliar.append(target_word)  # new line
                            auxiliar.append(antonimo)
        conjunto.append(auxiliar)

        for conjuntito in conjunto:
            if len(conjuntito) == 0:
                conjunto.remove(conjuntito)

    return conjunto
