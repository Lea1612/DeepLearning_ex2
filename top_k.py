import numpy as np
import sys


def similar_words(words_list_to_compare, voc_from_vocab_file):
    for word in words_list_to_compare:
        word_index = voc_from_vocab_file.index(word)
        vector = word_vectors[word_index]
        cos = np.full(len(word_vectors), -np.inf)
        for word_vectors_index, vec in enumerate(word_vectors):
            if word_index != word_vectors_index:
                cos[word_vectors_index] = cosine(vec, vector)
        write_in_file(word, voc_from_vocab_file, cos)


def cosine(vec, vector):
    return (vector.dot(vec)) / (np.linalg.norm(vec) * np.linalg.norm(vector))


def write_in_file(word, voc_from_vocab_file, res):
    print(f"{word} similar words: {[(voc_from_vocab_file[i], res[i]) for i in res.argsort()[-5:][::-1]]}")


if __name__ == '__main__':
    vocab_txt = sys.argv[1]
    word_vectors_txt = sys.argv[2]
    words_to_compare = ['dog', 'england', 'john', 'explode', 'office']
    word_vectors = np.loadtxt(word_vectors_txt)
    voc = open(vocab_txt).read().split('\n')
    similar_words(words_to_compare, voc)
