"""Script for runing tests."""


from principal import word_synset_construct

lis = ['house', 'car']

lis_out = word_synset_construct(list_of_words=lis)

print(lis_out)
