from principal import TER

embedding = TER(path="Embeddings/", type='Word2Vec')
embedding.filterWN()

print(len(embedding.words))
print(len(embedding.filtered_words))
