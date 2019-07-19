''' 
:return: None
'''

def filterWN(self):

    wn_lemmas = set(wn.all_lemma_names())
    for j in self.words:
        if j in wn_lemmas:
            self.filtered_words.append(j)

    self.filtered_words = list(set(self.filtered_words))
