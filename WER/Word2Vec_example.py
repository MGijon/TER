from principal import distance, random_pairs, load_embedding, return_vector, filter_WN



# Word2Vec
# ========

datos = load_embedding(path="../Embeddings/",
                       embeding_name="GoogleNews-vectors-negative300.bin.gz",
                       embedings_size=300,
                       type="Word2Vec")

print(return_vector(datos, type="Word2Vec", setOfWords=['house']))




##############################################
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
