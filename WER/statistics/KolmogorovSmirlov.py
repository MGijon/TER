# Scipy package
from scipy import stats


def KolmogorovSmirlov(data1=[], data2=[]):
    '''
    Compute the Kolmogorov-Smirlov statistics of the two given distributions,
    :param data1: array of values (distribution 1).
    :param data2: array of values (distribution 2).
    :return: D (float), p-value (float)
    '''
    return stats.ks_2samp(data1, data2)
