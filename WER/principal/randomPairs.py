""" """
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


def random_pairs(listOfWords, numberOfPairs):
    """random_pairs."""
    pairs = []

    for i in range(1, numberOfPairs - 1):
        secure_random = random.SystemRandom()
        pairs.append((secure_random.choice(listOfWords),
                      secure_random.choice(listOfWords)
                      ))

    return pairs


def random_pairs_list(arrayOfLists, numberOfPairs):
    """random_pairs_list"""
    pairs = []

    for sublist in arrayOfLists:
        for _ in len(0, len(sublist)):
            pairs.append((secure_random.choice(sublist),
                          secure_random.choice(sublist)
                          ))

    return pairs
