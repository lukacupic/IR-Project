class Document:

    def __init__(self, words, path, tf, tf_idf):
        self.words = words
        self.path = path
        self.tf = tf
        self.tf_idf = tf_idf

    def getWords(self):
        return self.words

    def setWords(self, words):
        self.words = words

    def getPath(self):
        return self.path

    def setPath(self, path):
        self.path = path

    def getTf(self):
        return self.tf

    def setTf(self, tf):
        self.tf = tf

    def getTfIdf(self):
        return self.tf_idf

    def setTfIdf(self, tf_idf):
        self.tf_idf = tf_idf
