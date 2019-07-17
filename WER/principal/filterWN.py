def filterWN(self):
    '''
    From self.words takes the words that are in WordNet and save them in
    self.filtered_words.
    :return: None
    '''
    self.logger.info('Starting filtrated with WordNet')

    wn_lemmas = set(wn.all_lemma_names())
    for j in self.words:
        if j in wn_lemmas:
            self.filtered_words.append(j)

    self.filtered_words = list(set(self.filtered_words))

    self.logger.info('Finished filtrated with WordNet')
