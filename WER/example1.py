from principal import WER

# load an Wor2Vec embedding
embedding = WER(path="Embeddings/", type="Word2Vec")
# filter by WordNet
embedding.filterWN()

print("The embedding has a vocabulary composed for a total of " + str(len(embedding.words)) + " words")
print("Just " + str(len(embedding.filtered_words)) + " of them are in WordNet")
