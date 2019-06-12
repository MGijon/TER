from principal import WER
import matplotlib.pyplot as plt

# load an Wor2Vec embedding
embedding = WER(path="Embeddings/", type="GloVe")
# filter by WordNet
embedding.filterWN()

# take synonims
embedding.synonymsFilteredWords(norma = 30)
# random words
embedding.randomFilteredWords(norma = 30)
# antonyms
embedding.antonymsFilteredWords(norma = 30)


## Synonyms Vs Antonyms
## --------------------
plt.figure(figsize=(20, 14))

ks, p = embedding.KolmogorovSmirlov(data1=embedding.synonimsDistribution, data2=embedding.antonimsDistribution)
_, bins, _ = plt.hist(embedding.synonimsDistribution, bins=50, normed=True, alpha=.3, label="synonyms")
_ = plt.hist(embedding.antonimsDistribution, bins=bins, alpha=0.3, normed=True, label="antonyms")

plt.title("Synonyms vs. Antonyms " + ' - ks: ' + str(ks) + ' - p value: ' + str(p))
plt.xlabel("values")
plt.legend()
plt.savefig("Results/EpsilonPonderateSynonymsAntonyms")


## Synonyms Vs Random
## ------------------

plt.figure(figsize=(20, 14))
ks2, p = embedding.KolmogorovSmirlov(data1=embedding.synonimsDistribution, data2=embedding.randomDistribution)
_, bins, _ = plt.hist(embedding.synonimsDistribution, bins=50, normed=True, alpha=.3, label="synonyms")
_ = plt.hist(embedding.randomDistribution, bins=bins, alpha=0.3, normed=True, label="not ralated words")


plt.title("Synonyms vs. not related words " + ' - ks: ' + str(ks2) + ' - p value: ' + str(p))
plt.xlabel("values")
plt.legend()
plt.savefig("Results/EpsilonPonderateSynonymsRandom")


## Antonyms Vs Random
## ------------------

plt.figure(figsize=(20, 14))
ks2, p = embedding.KolmogorovSmirlov(data1=embedding.antonimsDistribution, data2=embedding.randomDistribution)
_, bins, _ = plt.hist(embedding.antonimsDistribution, bins=50, normed=True, alpha=.3, label="antonyms")
_ = plt.hist(embedding.randomDistribution, bins=bins, alpha=0.3, normed=True, label="not ralated words")


plt.title("Antonyms vs. not related words " + ' - ks: ' + str(ks2) + ' - p value: ' + str(p))
plt.xlabel("values")
plt.legend()
plt.savefig("Results/EpsilonPonderateAntonymsRandom")
