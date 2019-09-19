"""Graphics pre-defined functions."""

# Mathplotlib
import matplotlib.pyplot as plt

def stackedBar(list_1, list_2, title_list1="", title_list2="", title="", xlabel="", ylabel="", savetitle=""):
    '''

    :param list_1:
    :param list_2:
    :param title_list1:
    :param title_list2:
    :param xlabel:
    :param ylabel:
    :param savetitle:
    :return:
    '''

    p1 = plt.bar(1, list_1)
    p2 = plt.bar(1, list_2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend((p1[0], p2[0]), (title_list1, title_list2))

    plt.savefig(savetitle)
    plt.show()

def twoHistograms(list_1, list_2, title_list1="", title_list2="", title="", xlabel="", ylabel="",
                      savetitle=""):
    '''

    :param list_1:
    :param list_2:
    :param title_list1:
    :param title_list2:
    :param title:
    :param xlabel:
    :param ylabel:
    :param savetitle:
    :return:
    '''
    plt.figure(figsize=(20, 14))

    s = Estadisticas()
    ks, p = s.KolmogorovSmirlov(data1=list_1, data2=list_2)

    _, bins, _ = plt.hist(list_1, bins=50, normed=True, alpha=.3, label=title_list1)
    _ = plt.hist(list_2, bins=bins, alpha=0.3, normed=True, label=title_list2)

    plt.title(title + ' - ks: ' + str(ks) + ' - p value: ' + str(p))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()

    plt.savefig(savetitle)
    plt.show()
