
# c imports
cimport cython
from ranker cimport *

import sys
from .trainer import load_model

if sys.version_info[0] == 3:
    PY3 = True
else:
    PY3 = False

cdef class QueryDocumentRelevance:
    def __cinit__(self, counts, total_docs):
        '''
        Load the model and construct the C++ class

        counts: the token -> (word count, document count) map from the corpus
        total_docs: total documents in the corpus
        '''
        # need to convert to bytes...
        if PY3:
            cnts = {}
            for k, v in counts.items():
                cnts[k.encode('utf-8')] = v
        else:
            cnts = counts

        self._qdr_ptr = new QDR(cnts, total_docs)

    def __dealloc__(self):
        del self._qdr_ptr

    def _to_utf8(self, tokens):
        return [token.encode('utf-8') for token in tokens]

    def score(self, document, query):
        '''
        Compute the query-document relevance scores

        document and query are tokenized lists of words
        '''
        # cython will handle the conversion for us...
        if PY3:
            return self._qdr_ptr.score(
                self._to_utf8(document), self._to_utf8(query))
        else:
            return self._qdr_ptr.score(document, query)

    def score_batch(self, document, queries):
        '''
        Compute the query-document relevance scores for a group of queries
            against a single document

        document is a list of tokenized words
        queries is a list of queries, each query is a list of tokenized words
        '''
        if PY3:
            q = [self._to_utf8(query) for query in queries]
            return self._qdr_ptr.score_batch(self._to_utf8(document), q)
        else:
            return self._qdr_ptr.score_batch(document, queries)

    def get_idf(self, word):
        if PY3:
            word = word.encode('utf-8')
        return self._qdr_ptr.get_idf(word)

    @classmethod
    def load_from_file(cls, inputfile):
        ndocs, counts = load_model(inputfile)
        ret = cls(counts, ndocs)
        return ret

