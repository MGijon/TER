'''
:return: 
'''

# NLTK package
from nltk.corpus import wordnet as wn


def filterWN(setOfWords):

    auxiliar = []

    wn_lemmas = set(wn.all_lemma_names())
    for word in setOfWords:
        if word in wn_lemmas:
            auxiliar.append(word)

    auxiliar = list(set(auxiliar))
    return auxiliar
