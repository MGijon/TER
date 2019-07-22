"""  """

def loadEmbeddingDict(name="saveEmbeddingWithoutName"):
    '''
    Load a dictionary, (word, representation_in_the_embedding)
    :param name: name of the file
    :return: Dictionary (str, Array of numbers)
    '''

    filename = name
    infile = open(filename, 'rb')
    data = plk.load(infile)
    infile.close()

    return (data)
