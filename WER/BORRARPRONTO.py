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





class WER(object):
    '''
    '''








    # eliminar parámetro all, no es necesario
    def randomDistances(self, words, number=5000, all=False, norma=1):
        '''

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
        '''

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

    def pureSynonyms(self):
        '''
        Compute the set of synonims, without distances
        :return: None
        '''
        conjunto = []
        for target_word in self.filtered_words:
            synsets = wn.synsets(target_word)
            for synset in synsets:
                auxiliar = []
                lemmas = synset.lemmas()
                numberSynom = len(lemmas)
                if numberSynom > 1:
                    for lemma in lemmas:
                        palabra = lemma.name()
                        if palabra in self.filtered_words:
                            auxiliar.append(palabra)
            conjunto.append(auxiliar)

            for conjuntito in conjunto:
                if len(conjuntito) == 0:
                    conjunto.remove(conjuntito)

        self.synonims = conjunto

    def pureAntonyms(self):
        '''
        Just compute the set of synonims, without distances
        :return: None
        '''

        conjunto = []
        for target_word in self.filtered_words:
            synsets = wn.synsets(target_word)
            for synset in synsets:
                auxiliar = []
                lemmas = synset.lemmas()
                numberSynom = len(lemmas)
                if numberSynom > 1:
                    for lemma in lemmas:
                        if lemma.antonyms():
                            antonimo = lemma.antonyms()[0].name()
                            if antonimo in self.filtered_words:
                                auxiliar.append(target_word)  # new line
                                auxiliar.append(antonimo)
            conjunto.append(auxiliar)

            for conjuntito in conjunto:
                if len(conjuntito) == 0:
                    conjunto.remove(conjuntito)

        self.antonims = conjunto

    def antonymsFilteredWords(self, norma = 1):
        '''
        Fills the antonimsDistribution array (from antonims set)
        :param norma:
        :return: None
        '''

        self.pureAntonyms() # self.antonims

        self.antonimsDistribution = self.randomDistancesList(self.antonims, norma = norma)

    def synonymsFilteredWords(self, norma=1):
        '''

        :param norma:
        :param number:
        :return: None
        '''

        group = []
        for target_word in self.filtered_words:
            synsets = wn.synsets(target_word)
            for synset in synsets:
                auxiliar = []
                lemmas = synset.lemmas()
                numberSynom = len(lemmas)
                if numberSynom > 1:
                    for lemma in lemmas:
                        word = lemma.name()
                        if word in self.filtered_words:
                            auxiliar.append(word)
            group.append(auxiliar)

            for littleGroup in group:
                if len(littleGroup) == 0:
                    group.remove(littleGroup)

        self.synonims = group
        self.synonimsDistribution = self.randomDistancesList(list=group, norma=norma)



    ########################################################
    def synonymsComplementary(self, norma=1, number=5000):
        '''
        Fill synonimsComplementary array and compute a random sample of their distribution
        :param norma:
        :param number:
        :return: None
        '''

        words_no_synomims = []
        for target_word in self.filtered_words:
            synsets = wn.synsets(target_word)
            for synset in synsets:
                lemas = synset.lemmas()
                numberSynom = len(lemas)
                if numberSynom == 1:
                    word = lemas[0].name()
                    if word in self.filtered_words:
                        words_no_synomims.append(p)

        words_no_synomims = list(set(words_no_synomims))
        self.synonimsComplementary = words_no_synomims

        self.synonimsDistributionComplementary = self.randomDistances(words=auxiliar,
                                                                      norma=norma,
                                                                      number=number)

    ########################################################


    def randomFilteredWords(self, norma=1, number=5000, all=False):
        '''

        :param norma:
        :param number:
        :return:
        '''

        if all:
            self.randomDistribution = self.randomDistances(words=self.words,
                                                           norma=norma,
                                                           number=number)
        else:
            self.randomDistribution = self.randomDistances(words=self.filtered_words,
                                                           norma=norma,
                                                           number=number)


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

    def notVocabularyDistribution(self, norma = 1):
        '''
        :param norma:
        :return: Return array of floats with the distributions of the words that
                 are not in the filtered group.
        '''

        conjunto_palabras = set(self.words) - set(self.filtered_words)

        return self.randomDistances(norma=norma, words=list(conjunto_palabras), number=10000)

    def wordSynsetConstruct(self):
        '''
        Take the words save in the array self.filtered_words and fill the array
        self.words_synsets with arrays [word, name of synsets it belongs].
        :return: None
        '''

        if len(self.filtered_words) == 0:
            self.filterWN()

        lista = []
        for j in self.words:
            words_synsets = wn.synsets(j)
            aux = []
            for i in words_synsets:
                aux.append(i.name())
            lista.append((j, aux))

        self.wordSynset = lista

    @staticmethod
    def clearArrayOfArrays(data=[]):
        '''

        :param data:
        :return:
        '''
        logger.info("Starting clearArrayOfArrays")
        newData = []
        for i in data:
            auxiliar = []
            for j in i:
                if j != 0.0:
                    auxiliar.append(j)
            if len(auxiliar) != 0:
                newData.append(auxiliar)

        self.logger.info("Ended clearArrayOfArrays")
        return (newData)

    @staticmethod
    def arrayOfArraysToArray(data=[]):
        '''

        :param data:
        :return:
        '''
        logger.info("Starting arrayOfArraysToArray")
        resoult = []
        for i in data:
            for j in i:
                resoult.append(j)

        self.logger.info("Ended arrayOfArraysToArray")
        return (list(set(resoult)))

    def saveWords(self, name="saveWordsWithoutName"):
        '''
        Save lists of words in a pickle file
        :param name:
        :return: None
        '''

        self.logger.info('Starting saveWords')
        try:
            filename = name
            outfile = open(filename, 'wb')
            plk.dump(self.words, outfile)
            outfile.close()
        except Exception as e:
            print(e)
            pass
        self.looger.info('Finished saveWords')

    @staticmethod
    def loadWords(name="saveWordsWithoutName"):
        '''
        Load an array of words in pickle format
        :param name: name of the file
        :return: Array of strings
        '''
        self.logger.info("Starting laodWords")
        filename = name
        infile = open(filename, 'rb')
        data = plk.load(infile)
        infile.close
        self.looger.info("Ended loadWords")
        return (data)

    @staticmethod
    def saveData(option=1, name='saveDataWihoutName', data=[]):
        '''
        Save numerical data from the calculations in python or numpy formats
        :param option:
        :param name:
        :param data:
        :return: None
        '''

        if option == 1:

            filename = name
            outfile = open(filename, 'wb')
            plk.dump(data, outfile)
            outfile.close()

        elif option == 2:

            outfile = TemporaryFile()
            np.save(outfile, data)

        else:
            print('Error, please read the documentation of this function')
            pass

    @staticmethod
    def loadData(name="saveDataWithoutName"):
        '''
        Load a numeric array in pickle format
        :param name: name of the file
        :return: Array of numbers
        '''

        filename = name
        infile = open(filename, 'rb')
        data = plk.load(infile)
        infile.close()

        return (data)

    def saveEmbedding(self, name="saveEmbeddingWithoutName"):
        '''
        Save the words and their representations in a dictionary using pickle format
        :param name:
        :return: None
        '''

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

    @staticmethod
    def loadEmbeddingDict(name="saveEmbeddingWithoutName"):
        '''
        Load a dictionary, (word, representation_in_the_embedding)
        :param name: name of the file
        :return: Dictionary (str, Array of numbers)
        '''

        filename = name
        infile = open(filename, 'rb')
        data = plk.load(infile)
        infile.close()

        return (data)
