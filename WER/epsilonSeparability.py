from principal import WER


# load an Wor2Vec embedding
embedding = WER(path="Embeddings/", type="Word2Vec")
# filter by WordNet
#embedding.filterWN()

vectors = embedding.returnVector(setOfWords = ['house'])
print(vectors)
#ahora debemos trabajar con las representaciones vextoriales muhahahahha
