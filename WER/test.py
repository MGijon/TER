from principal import loadEmbedding, returnVector, filterWN

import os

## SEMANA 2
## filterWN .- poder filtrar el vocabulario con wordnet
## Hacer funcionar la función de distancias
## funcion para tomar parejas aleatorias con una distancia dada


## ---------------------------------------------------------------------------------------------------
## !! printar histogramsas de vectores podría ser una buena idea de comprobar la teoría!!!!
    ## usar ks  sobre las familias de sinónimos para ver si coninciden o no muhahaha
    ## antes  debo teorizar cuál debe de ser el threshold para estas distribuciones discretas
## NO ME PREOCUPARË AHORA POR la DOCUMENTACIÖN MÄS ALLÄ DE NOTAS BÄSICAS PARA MÍ
## por ahora coloco todas las dependencias, borraré las innecesarias en el futuro!!


# Word2Vec
datos = loadEmbedding.loadEmbedding(path="../Embeddings/",
                                    embeding_name="GoogleNews-vectors-negative300.bin.gz",
                                    embedings_size=300,
                                    type="Word2Vec")

#print(returnVector.returnVector(datos, type="Word2Vec", setOfWords=['house']))



#print([file for file in os.listdir('../Embeddings/')])
# GloVe dimension 300
#size = 300
#datos = loadEmbedding.loadEmbedding(path="../Embeddings/",
#                                    embeding_name='glove.6B.300d.txt',
#                                    embedings_size=size,
#                                    type="GloVe")

#print(returnVector.returnVector(datos, type="GloVe", setOfWords=['house']))

palabras = datos['words']
print(len(palabras))

palabras = filterWN.filterWN(palabras)
print(len(palabras))
print(palabras[0:20])

#print(size)

#print(len(datos['words']))





## SEMANA 2
