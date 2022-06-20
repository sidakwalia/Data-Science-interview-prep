import numpy

def read_glove_vectors(glove_pretrained_file):
    word_embeddings_matrix = {}

    f = open(glove_pretrained_file, encoding='utf')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = numpy.asarray(values[1:], dtype='float32')
        word_embeddings_matrix[word] = coefs
    f.close()

    return word_embeddings_matrix