import numpy as np
from collections import Counter


class Method:

    def __init__(self):
        pass

    def getVector(self, vector):
        pass

    def getIdf(self, tfs, documents):
        return np.ones(len(self.corpusVector))

    def setCorpusVector(self, corpusVector):
        self.corpusVector = corpusVector


class BitVector(Method):

    def __init__(self):
        super().__init__()

    def getVector(self, words):
        bv = []
        for w in self.corpusVector:
            bv.append(1 if w in words else 0)
        return np.array(bv)


class Tf(Method):

    def __init__(self):
        super().__init__()

    def getVector(self, words):
        tf = []
        counts = Counter(words)
        for w in self.corpusVector:
            tf.append(counts[w])
        return np.array(tf)


class TfIdf(Method):

    def __init__(self):
        super().__init__()

    def getVector(self, words):
        tf = []
        counts = Counter(words)
        for w in self.corpusVector:
            tf.append(counts[w])
        return np.array(tf)

    def getIdf(self, tfs, documents):
        corpusLen = len(self.corpusVector)
        docsLen = len(documents)

        idf = []
        for i in range(corpusLen):
            count = 0
            for tf in tfs:
                if tf[i] != 0:
                    count = count + 1
            idf.append(np.log10((docsLen + 1) / float(count)))
        return np.array(idf)
