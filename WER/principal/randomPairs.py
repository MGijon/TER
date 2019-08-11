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


def randomPairs(listOfWords, numberOfPairs):

    pairs = []

    for i in range(1, numberOfPairs - 1):
        secure_random = random.SystemRandom()
        pairs.append((secure_random.choice(listOfWords),
                      secure_random.choice(listOfWords)))


    return pairs


def randomPairsList(arrayOfLists, numberOfPairs):

    pairs = []

    for sublist in arrayOfLists:
        for _ in len(0, len(sublist)):
            # esto debe de ser mejorado y bastante
            pairs.append(secure_random.choice(sublist),
                          secure_random.choice(sublist)))

    return pairs
