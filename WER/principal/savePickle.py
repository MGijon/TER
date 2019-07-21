""" Save lists of words in a pickle file
:param name:
:param element:
:return: None
"""

def savePickle(self, name, element):
    try:
        filename = name
        outfile = open(filename, 'wb')
        plk.dump(element, outfile)
        outfile.close()
    except Exception as e:
        print(e)
        pass
