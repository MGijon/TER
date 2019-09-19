"""Statistical tests."""

# Scipy
from scipy import stats


def KolmogorovSmirlov(data1=[], data2=[]):
    '''
    Compute the Kolmogorov-Smirlov statistics of the two given distributions,
    :param data1: array of values (distribution 1).
    :param data2: array of values (distribution 2).
    :return: D (float), p-value (float)
    '''
    return stats.ks_2samp(data1, data2)

def Lilliefors(distribution1, distribution2):
    """Lilliefors test."""
    # TODO
    return None

def ShapiroWilk(distribution1, distribution2):
    """Shapiro-Wilk test."""
    # TODO
    return None

def AndersonDarling(distribution1, distribution2):
    """Anderson-Darling test."""
    # TODO
    return None
