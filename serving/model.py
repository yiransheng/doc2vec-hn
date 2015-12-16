from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec

from gensim.models.doc2vec import train_document_dbow
import cPickle as pickle


from numpy import load as np_load, zeros, random, mean, sum as np_sum, add as np_add, concatenate, \
    repeat as np_repeat, array, float32 as REAL, empty, ones, memmap as np_memmap, \
    sqrt, newaxis, ndarray, dot, vstack, dtype, divide as np_divide


class FakeModel:
    def __init__(self, path, size=100):
        self.vector_size = size
        self.layer1_size = size
        self.path = path
        self.syn0 = None
        self.syn1neg = None
        self.vocab = None
        self.cum_table = None
        self.hs = 0
        self.negative = 8
        self.random = random
        self.neg_labels = zeros(self.negative + 1, dtype=REAL)
        self.neg_labels[0] = 1.

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, a,b,c):
        pass

    def start(self):
        self.syn0 = np_load("%s.syn0.npy" % self.path) 
        self.syn1neg = np_load("%s.syn1neg.npy" % self.path) 
        with open("%s.vocab" % self.path, 'r') as f:
            self.vocab = pickle.load(f) 
        self.cum_table = np_load("%s.cum_table.npy" % self.path)
        

    def rand_vec(self):
        return (random.rand(self.vector_size) - 0.5) / self.vector_size

    def lookup_words(self, positive, negative):
        ipos = [self.vocab[w].index for w in positive if w in self.vocab]
        ineg = [self.vocab[w].index for w in negative if w in self.vocab]
        if len(ineg) == 0:
            return np_sum(self.syn0[ipos], axis=0) / len(ipos)
        return (np_sum(self.syn0[ipos], axis=0) - np_sum(self.syn0[ineg], axis=0)) / (len(ipos) + len(ineg))

    def lookup_word(self, word):
        if word not in self.vocab:
            return None
        i = self.vocab[word].index
        return self.syn0[i]

    def infer_vector(self, words, alpha=0.1, min_alpha=0.0001, steps=5):
        """
        Infer a vector for given post-bulk training document.
        Document should be a list of (word) tokens.
        """
        doctag_vectors = empty((1, self.vector_size), dtype=REAL)
        doctag_vectors[0] = self.rand_vec().astype(REAL) 
        doctag_locks = ones(1, dtype=REAL)
        doctag_indexes = [0]

        work = zeros(self.layer1_size, dtype=REAL)

        for i in range(steps):
            train_document_dbow(self, words, doctag_indexes, alpha, work,
                                    learn_words=False, learn_hidden=False,
                                    doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            alpha = ((alpha - min_alpha) / (steps - i)) + min_alpha

        return doctag_vectors[0]

