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


class IdentityTransform(Transform):

    def __init__(self):
        super().__init__()

    def transform(self, vector):
        return vector


class BM25Transform(Transform):

    def __init__(self, k):
        super().__init__()
        self.k = k

    def transform(self, vector):
        return ((self.k + 1) * vector) / (self.k + vector)


class LogTransform(Transform):

    def __init__(self):
        pass

    def transform(self, vector):
        return np.log10(1 + vector)
