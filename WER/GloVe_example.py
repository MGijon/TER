"""GloVe (dimension 300)."""

# WER
from principal import distance, random_pairs, random_pairs_list, load_embedding, return_vector, filter_WN, synonyms, save_pickle


size = 300
# loading the data
datos = load_embedding(path="../Embeddings/",
                       embeding_name='glove.6B.300d.txt',
                       embedings_size=size,
                       type="GloVe")

# We can obtain the vectorial representation of a set of words like this
#print(return_vector(datos, type="GloVe", setOfWords=['house']))

words = datos['words']
#print(len(words))


## EXPERIMENT 1:
## =============

# (1) Take the set of synonyms and the set of words that are in wordnet
# (2) Take random pairs of words in each set (synonym words go toghether)
# (3) Measure the distances between them
# (4) Keep the data to future analysis

# (1)
# ---

words_filtered = filter_WN(words) # just the vocabulary in WordNet

synon = synonyms(words=words_filtered) # return a list of arrays wiht synonym words

# (2)
# ---

words_filtered_random_pairs = random_pairs(listOfWords=words_filtered, numberOfPairs=500)

random_synonyms = random_pairs_list(arrayOfLists=synon)

# (3)
# ---

distances_random_pairs = []
for pair in words_filtered_random_pairs:
    representations_in_the_embedding = return_vector(datos, type="GloVe", setOfWords=[pair[0], pair[1]])
    dist = distance(vector1=representations_in_the_embedding[0],
                    vector2=representations_in_the_embedding[1],
                    norm=2)
    distances_random_pairs.append(dist)

distances_synonyms_pairs = []
for pair in random_synonyms:
    representations_in_the_embedding = return_vector(datos, type="GloVe", setOfWords=[pair[0], pair[1]])
    dist = distance(vector1=representations_in_the_embedding[0],
                    vector2=representations_in_the_embedding[1],
                    norm=2)
    distances_synonyms_pairs.append(dist)

# (4)
# ---

save_pickle(name='random_words_GloVe_example', element=distances_random_pairs)
save_pickle(name='synonyms_GloVe_example', element=distances_synonyms_pairs)
