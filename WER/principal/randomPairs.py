"""

"""
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

def randomPairs(listOfWords):
    return None





'''
# eliminar parámetro all, no es necesario
def randomDistances(self, words, number=5000, all=False, norma=1):


    :param words: array, list of words
    :param number: number od samples if all = False
    :param all: boolean, if True then take as many random distances as elements
                has the set.
    :param norma:
    :return: distances array
    '''

    self.logger.info("Start taking random distances")
    distances = []

    pairs = []

    # GloVe
    # =====
    if self.type == 1:
        if all:
            number = len(words)

        for i in range(1, number - 1):
            secure_random = random.SystemRandom()
            pairs.append((secure_random.choice(words), secure_random.choice(words)))

        for j in range(0, number - 1):
            try:
                distance = self.norm(vector=self.embeddings_index[words[j]],
                                     vector2=self.embeddings_index[words[j + 1]],
                                     norma=norma)
                distances.append(distance)
            except Exception as e:
                print (e)
                distances.append(0)
                pass

    # Word2Vec
    # ========
    elif self.type == 2:
        if all:
            number = len(words)

        for i in range(1, number - 1):
            secure_random = random.SystemRandom()
            pairs.append((secure_random.choice(words), secure_random.choice(words)))

        for j in range(0, number - 1):
            try:
                distance = self.norm(vector=self.model.get_vector(words[j]),
                                     vector2=self.model.get_vector(words[j + 1]),
                                     norma=norma)
                distances.append(distance)
            except Exception as e:
                print (e)
                distances.append(0)
                pass

    else:
        pass

    return distances

def randomDistancesList(self, list, norma=1):


    :param list:
    :param number:
    :param all:
    :param norma:
    :return: array with distances between elements of the sets in the given
             set
    '''

    distances = []

    # GloVe
    # =====
    if self.type == 1:
        for synonims_set in list:
            for i in range(0, len(synonims_set)):
                if i + 1 < len(synonims_set):
                    try:
                        distance = self.norm(vector=self.embeddings_index[synonims_set[i]],
                                             vector2=self.embeddings_index[synonims_set[i + 1]],
                                             norma=norma)
                        distances.append(distance)
                    except Exception as e:
                        print(e)
                        distances.append(0)
                        pass

    # Word2Vec
    # ========
    elif self.type == 2:
        for synonims_set in list:
            for i in range(0, len(synonims_set)):
                if i + 1 < len(synonims_set):
                    try:
                        distance = self.norm(vector=self.model.get_vector(synonims_set[i]),
                                             vector2=self.model.get_vector(synonims_set[i + 1]),
                                             norma=norma)
                        distances.append(distance)
                    except Exception as e:
                        print(e)
                        distances.append(0)
                        pass
    else:
        pass


    self.logger.info('Finished random distances in the array of arrays')

    return distances



'''
