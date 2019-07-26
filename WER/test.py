from principal import loadEmbedding

import os

## SEMANA 1
## - Cargar el embedding con las funciones del paquete,
## - poder elegir el tipo de embedding y la ruta en la que lo almacenamos con un
##   dos stings diferentes
##  - Obtener representaciones vectoriales qye se impriman por pantalla de cada una de los vectoriales

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

# GloVe dimension 300
#size = 300
#datos = loadEmbedding.loadEmbedding(path="../Embeddings/",
#                                    embeding_name='glove.6B.300d.txt',
#                                    embedings_size=size,
#                                    type="GloVe")

#print(size)

#print(len(datos['words']))




#print([print(file) for file in os.listdir('../Embeddings/')])
## SEMANA 2
