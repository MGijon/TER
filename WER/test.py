from principal import distance, random_pairs, load_embedding, return_vector#, #filterWN,

import os


# Word2Vec
# ========

'''
datos = load_embedding(path="../Embeddings/",
                       embeding_name="GoogleNews-vectors-negative300.bin.gz",
                       embedings_size=300,
                       type="Word2Vec")

print(return_vector(datos, type="Word2Vec", setOfWords=['house']))

'''

#print([file for file in os.listdir('../Embeddings/')])
# GloVe (dimension 300)
# =====================
size = 300
datos = load_embedding(path="../Embeddings/",
                       embeding_name='glove.6B.300d.txt',
                       embedings_size=size,
                       type="GloVe")

print(return_vector(datos, type="GloVe", setOfWords=['house']))

#palabras = datos['words']
#print(len(palabras))

#palabras = filterWN.filterWN(palabras)
#print(len(palabras))
#print(palabras[0:20])

#print(size)

#print(len(datos['words']))
#vector1 = [0 ,1, 2, 3, 4, 5 ,-56]
#vector2 = [32 ,21, 22, 123, -34, 5 ,56]

#valor = distance.distance(vector1=vector1, vector2=vector2, norm=28)

#palabras_test =['hola', 'cabron', 'malnacido', 'desgraciado', 'felon', 'capullo',
#                'comepollas', 'tragasables']

#devueltas = randomPairs.random_pairs(listOfWords=palabras_test,
#                                    numberOfPairs=10)

#devueltas = random_pairs(listOfWords=palabras_test,
#                                    numberOfPairs=10)


#print(devueltas)
## SEMANA 2
