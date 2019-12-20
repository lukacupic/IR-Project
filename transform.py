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


class LogTransform(Transform):

    def __init__(self):
        pass

    def transform(self, vector):
        return np.log10(1 + vector)

    def getName(self):
        return "log"
