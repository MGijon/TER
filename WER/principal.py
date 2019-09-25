"""Principal functions of the package."""

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

# EMBEDDING_MANIPULATION
# ======================
def load_embedding(embeding_name='', embedings_size=300, path='Embeddings/',  type='GloVe'):
    """Load a pretrained embedding (GloVe or Word2Vec)."""
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
            indexes = {}
            with open(os.path.join(path, embeding_name)) as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    indexes[word] = coefs
                dict['embeddings_index'] = indexes
                f.close()


            dict['words'] = list(indexes.keys())


        # Word2Vec
        # --------
        elif type == 2 or allowed_types[type] == 2:
            dict['dimension'] = embedings_size
            dict['embeding_name'] = embeding_name
            dict['model'] = gensim.models.KeyedVectors.load_word2vec_format(
                                            path + embeding_name,
                                            binary=True)
            dict['words'] = list(dict['model'].vocab)

        else:
            print('Fatal error loading the embedding')


        return dict

    except Exception as e:
        print(e)

def return_vector(embedding_dictionay, type, setOfWords):
    """ """
    vectorsArray = []
    type = type.lower()
    allowed_types = {'glove':1,
                     'word2vec':2}

    try:
        # GloVe
        if allowed_types[type] == 1:
            for word in setOfWords:
                vectorsArray.append(embedding_dictionay['embeddings_index'][word])

        # Word2Vec
        # ========
        elif allowed_types[type] == 2:
            for word in setOfWords:
                vectorsArray.append(embedding_dictionay['model'].get_vector(word))
    except:
        message = 'Sorry, in this list of words there is at least one word that is not in the vocabulary of the embedding'
        print(message)
        self.logger.info('Failed returnVector function - some word not in vocabulary\n')
        pass # to let the programm continue

    return vectorsArray

# DISTANCES
# =========
def distance(vector1, vector2, norm=1):
    ''' Compute the distance between two vector1s under the selected norm.
    :param vector1: (array, floats) self-explanatory.
    :param vector2: (array, floats) self-explanatory.
    :param norm: distance
    :return: value of the distence (under the selected norm) between the two vector1s
    '''
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

# SAMPLING
# ========
# this two will become one
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
##################################

# SEMANTIC PART
# =============
def filter_WN(setOfWords):
    """ """
    auxiliar = []

    wn_lemmas = set(wn.all_lemma_names())
    for word in setOfWords:
        if word in wn_lemmas:
            auxiliar.append(word)

    auxiliar = list(set(auxiliar))
    return auxiliar

def synonyms(self, words):
    '''
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
                    palabra = lemma.name()
                    if palabra in words:
                        auxiliar.append(palabra)
        conjunto.append(auxiliar)

        for conjuntito in conjunto:
            if len(conjuntito) == 0:
                conjunto.remove(conjuntito)

    return conjunto

def antonyms(self, words):
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

# UTILITIES
# =========
def array_of_arrays_to_array(data):
    '''

    :param data:
    :return:
    '''
    resoult = []
    for i in data:
        for j in i:
            resoult.append(j)

    return (list(set(resoult)))

def save_embedding(self, name, words, type):
    '''
    Save the words and their representations in a dictionary using pickle format
    :param name:
    :return: None
    '''

    ### RECONSTRUIR TODO ESTO
    data = {}
    for i in self.words:
        data[i] = self.embeddings_index[i]

    try:
        filename = name
        outfile = open(filename, 'wb')
        plk.dump(data, outfile)
        outfile.close()
    except Exception as e:
        print(e)
        pass

def save_pickle(self, name, element):
    """ Save lists of words in a pickle file
    :param name:
    :param element:
    :return: None
    """
    try:
        filename = name
        outfile = open(filename, 'wb')
        plk.dump(element, outfile)
        outfile.close()
    except Exception as e:
        print(e)
        pass

################################################################################

## TODO:  test it
def word_synset_construct(self, list_of_words):
    '''
    Given a list of words, it returns a list (word, other related word)
    :return:
    '''
    lista = []
    for word in list_of_words:
        words_synsets = wn.synsets(word)
        aux = []
        for i in words_synsets:
            aux.append(i.name())
        lista.append((word, aux))

    return (lista)





################################################################################


def distancesBetweenSet(self, norma=1, words=[]):
    '''
    Compute the distances between a word (the element 0 in the array) and a set of words.
    :param norma:
    :param words:
    :return: Array of arrays of words QUE PARARA SI EL ARRAY ES DE UN SOLO MODELO
    '''

    result = []

    # GloVe
    # =====
    if self.type == 1:
        for j in words:
            aux = []
            for i in j[1:]:
                inicial = self.embeddings_index[i[0]]
                try:
                    valor = self.norm(vector=inicial,
                                      vector2=self.embeddings_index[i],
                                      norma=norma)
                    aux.append(valor)
                except KeyError:
                    pass

            result.append(aux)

    # Word2Vec
    # ========
    elif self.type == 2:
        for j in words:
            aux = []
            for i in j[1:]:
                inicial = self.model.get_vector[i[0]]
                try:
                    valor = self.norm(vector=inicial,
                                      vector2=self.model.get_vector[i],
                                      norma=norma)
                    aux.append(valor)
                except KeyError:
                    pass

            result.append(aux)

    else:
        pass


    return (result)



def returnSinonyms(self, word):
    '''
    Returns and array of synonims of the word passed (based on WordNet)
    :param word: word whose synonims we want
    :return: array of synonims
    '''

    auxiliar = []

    for i in wn.synsets(word):
        vector_auxiliar = []
        for j in [x.name() for x in i.lemmas()]:
            vector_auxiliar.append(j)

        if len(vector_auxiliar) > 1:
            auxiliar.append(list(set(vector_auxiliar)))

    return auxiliar
