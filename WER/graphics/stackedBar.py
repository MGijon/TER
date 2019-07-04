# Seaborn package
#import seaborn as sns
# Mathplotlib package
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
