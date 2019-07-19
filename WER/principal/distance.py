''' Compute the distance between two vectors under the selected norm.
:param vector: (array, floats) self-explanatory.
:param vector2: (array, floats) self-explanatory.
:param norm: distance
:return: value of the distence (under the selected norm) between the two vectors
'''
def distance(self, vector, vector2, norm=1):

    if norm == 1 or norm is "euclidean":

        calculo = vector
        for i in range(0, len(vector)):
            calculo[i] = calculo[i] - vector2[i]
        suma = 0
        for i in calculo:
            suma += np.power(i, 2)
        value = np.sqrt(suma)

    elif norm == 2 or norm is "cosine":
        value = scipy.spatial.distance.cosine(vector, vector2)

    elif norm == 3 or norm is "cityblock":
        value = scipy.spatial.distance.cityblock(vector, vector2)

    elif norm == 4 or norm is "l1":
         value = np.linalg.norm((vector - vector2), ord=1)

    elif norm == 7 or norm is "chebyshev":
        value = scipy.spatial.distance.chebyshev(vector, vector2)

    elif norm == 8 or norm is "minkowski":
        value = scipy.spatial.distance.minkowski(vector, vector2)

    elif norm == 9 or norm is "sqeuclidean":
        value = scipy.spatial.distance.sqeuclidean(vector, vector2)

    elif norm == 10 or norm is "jensenshannon":
        _P = vector / norm(vector, ord=1)
        _Q = vector2 / norm(vector2, ord=1)
        _M = 0.5 * (_P + _Q)
        value =  0.5 * (entropy(_P, _M) + entropy(_Q, _M))

    elif norm == 12 or norm is "jaccard":
        sklearn.metrics.jaccard_similarity_score(vector, vector2)

    elif norm == 13 or norm is "correlation":
        value = scipy.spatial.distance.correlation(vector, vector2)

    elif norm == 14 or norm is "braycurtis":
        value = scipy.spatial.distance.braycurtis(vector, vector2)

    elif norm == 15 or norm is "canberra":
        value = scipy.spatial.distance.canberra(vector, vector2)

    elif norm == 16 or norm is "kulsinski":
        value = scipy.spatial.distance.cdis(vector, vector2)

    elif norm == 17 or norm is "max5":
        # take the sum of the 5 maximun difference dimensions
        v = vector2 - vector
        v2 = [abs(x) for x in v]
        aux = heapq.nlargest(5, v2)
        value = sum(aux)

    elif norm == 18 or norm is "max10":
        # take the sum of the 10 maximun difference dimensions
        v = vector2 - vector
        v2 = [abs(x) for x in v]
        aux = heapq.nlargest(10, v2)
        value = sum(aux)


    elif norm == 19 or norm is "max25":
        # take the sum of the 25 maximun difference dimensions
        v = vector2 - vector
        v2 = [abs(x) for x in v]
        aux = heapq.nlargest(25, v2)
        value = sum(aux)

    elif norm == 20 or norm is "max50":
        # take the sum of the 50 maximun difference dimensions
        v = vector2 - vector
        v2 = [abs(x) for x in v]
        aux = heapq.nlargest(50, v2)
        value = sum(aux)

    elif norm == 21 or norm is "max100":
        # take the sum of the 100 maximun difference dimensions
        v = vector2 - vector
        v2 = [abs(x) for x in v]
        aux = heapq.nlargest(100, v2)
        value = sum(aux)

    elif norm == 28:
        non_sing_changes = 0

        for i in range(0, len(vector)):
            if vector[i] >= 0 and vector2[i] >= 0:
                non_sing_changes += 1
            if vector[i] < 0 and vector2[i] < 0:
                non_sing_changes += 1

        value = len(vector) - non_sing_changes

    elif norm == 29:
        epsilon = 0
        for coordinate in range(0, len(vector)):
            auxiliar = abs(vector[coordinate] - vector2[coordinate])
            if auxiliar > epsilon:
                epsilon = auxiliar
        value = epsilon

    elif norm == 30:
        epsions = 0
        for coordinate in range(0, len(vector)):
            epsions += abs(vector[coordinate] - vector2[coordinate])
        value = epsions / len(vector)


    elif norma == 31:
        differenceVector = [abs(vector[i] - vector2[i]) for i in range(0, len(vector))]
        value = differenceVector

    else:
        pass

    return value
