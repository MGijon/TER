from principal import distance, random_pairs, load_embedding, return_vector, filter_WN

# GloVe (dimension 300)
# =====================

size = 300
# loading the data
datos = load_embedding(path="../Embeddings/",
                       embeding_name='glove.6B.300d.txt',
                       embedings_size=size,
                       type="GloVe")

print(return_vector(datos, type="GloVe", setOfWords=['house']))

words = datos['words']
#print(len(words))

# reduce the number of words by taking just the ones in WordNet
words_filtered = filter_WN(words)
#print(len(words_filtered))
#print(words_filtered[0:20])
