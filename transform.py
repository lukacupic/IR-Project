import numpy as np


class Transform:

    def __init__(self):
        pass

    def transform(self, vector):
        pass

    def transformDocuments(self, docs):
        for d in docs:
            tf = self.transform(d.getTf())
            d.setTf(tf)

    def getName(self):
        pass

    def setDocuments(self, docs):
        self.docs = docs


class IdentityTransform(Transform):

    def __init__(self):
        super().__init__()

    def transform(self, vector):
        return vector

    def getName(self):
        return "identity"


class BM25Transform(Transform):

    def __init__(self, k):
        super().__init__()
        self.k = k

    def transform(self, vector):
        return ((self.k + 1) * vector) / (self.k + vector)

    def getName(self):
        return "bm25"


class BM25OkapiTransform(Transform):

    def __init__(self, k, b):
        super().__init__()
        self.k = k
        self.b = b

    def transform(self, vector):
        denom = vector + self.k * (1 - self.b + self.b * len(vector) / self.avdl)
        return ((self.k + 1) * vector) / denom

    def calculateAvdl(self):
        avdl = 0
        for d in self.docs:
            avdl = avdl + len(d.getWords())
        self.avdl = avdl / float(len(self.docs))

    def getName(self):
        return "bm25-okapi"

    def setDocuments(self, docs):
        super().setDocuments(docs)
        self.calculateAvdl()


class LogTransform(Transform):

    def __init__(self):
        pass

    def transform(self, vector):
        return np.log10(1 + vector)

    def getName(self):
        return "log"
