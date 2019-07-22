""" """

def saveEmbedding(self, name="", words, type):
    '''
    Save the words and their representations in a dictionary using pickle format
    :param name:
    :return: None
    '''


    ### RECONSTRUIR TODO ESTO
    data = {}
    for i in self.words:
        data[i] = self.embeddings_index[i]

    try:
        filename = name
        outfile = open(filename, 'wb')
        plk.dump(data, outfile)
        outfile.close()
    except Exception as e:
        print(e)
        pass
