import os
from collections import Counter

import numpy as np
from tika import parser

from preprocess import Preprocessor

filename = "tf_idf.pickle"
preprocessor = Preprocessor()


def toBitVector(d, corpusVector):
    bv = []
    for w in corpusVector:
        bv.append(1 if w in d else 0)
    return np.array(bv)


def toTfVector(d, corpusVector):
    tf = []
    counts = Counter(d)
    for w in corpusVector:
        tf.append(counts[w])
    return np.array(tf)


def toTfVectorDoc(d, corpusVector):
    tf = []
    counts = Counter(d[0])
    for w in corpusVector:
        tf.append(counts[w])
    d.append(np.array(tf))


def identity(vector, k):
    return vector


def logTransform(vector, k):
    return np.log10(1 + vector)


def bm25Transform(vector, k):
    return ((k + 1) * vector) / (k + vector)


def createTfVectors(documents, corpusVector, transform=None, k=None):
    for d in documents:
        toTfVectorDoc(d, corpusVector)
        if transform is not None:
            d[2] = transform(d[2], k)


def createIdfVector(corpusVector, tfs, documents):
    corpusLen = len(corpusVector)
    docsLen = len(documents)

    idf = []
    for i in range(corpusLen):
        count = 0
        for tf in tfs:
            if tf[i] != 0:
                count = count + 1
        idf.append(np.log10((docsLen + 1) / float(count)))
    return np.array(idf)


def readDataset(path):
    corpusVector = set()
    documents = []
    counter = 0
    categories = []

    for root, subdirs, files in os.walk(path):
        if counter != 0:
            categories.append([root, len(files)])

        counter = counter + 1
        for filename in files:
            filePath = os.path.join(root, filename)

            if not filePath.endswith('.pdf') and not filePath.endswith('.txt'):
                continue

            raw = parser.from_file(filePath)
            data = raw["content"].lower()

            words = preprocessor.preprocess(data)

            corpusVector.update(words)
            documents.append([words, filePath])

    return corpusVector, documents, categories


def isRelevant(document, top_n):
    for d in top_n:
        if document[1] == d[1]:
            return True
    return False


def evaluate(docs, path, top_n):
    print(top_n)
    matrix = [[0, 0], [0, 0]]

    for d in docs:
        relevant = isRelevant(d, top_n)

        if path in d[1] and relevant:
            matrix[0][0] = matrix[0][0] + 1

        if path not in d[1] and relevant:
            matrix[0][1] = matrix[0][1] + 1

        if path in d[1] and not relevant:
            matrix[1][0] = matrix[1][0] + 1

        if path not in d[1] and not relevant:
            matrix[1][1] = matrix[1][1] + 1
    return matrix


def computeScores(matrix):
    tp = matrix[0][0]
    fp = matrix[0][1]
    tn = matrix[1][0]
    fn = matrix[1][1]

    acc = (tp + tn) / np.float64(tp + tn + fp + fn)
    pre = tp / np.float64(tp + fp)
    rec = tp / np.float64(tp + fn)
    f1 = 2 * pre * rec / np.float64(pre + rec)

    print("Accuracy:  %.4f" % acc)
    print("Precision: %.4f" % pre)
    print("Recall:    %.4f" % rec)
    print("F1:        %.4f" % f1)


def main():
    corpusVector, documents, categories = readDataset('./dataset-small')

    if not os.path.exists(filename):
        createTfVectors(documents, corpusVector, transform=bm25Transform, k=0.1)
        tfs = [row[2] for row in documents]
        idf = createIdfVector(corpusVector, tfs, documents)

        for d in documents:
            d[2] = d[2] * idf
    # pickle.dump([tfs, idf, tf_idfs], open(filename, "wb"))
    else:
        # tfs, idf, tf_idfs = pickle.load(open(filename, "rb"))
        pass

    for c in categories:
        query = os.path.basename(c[0])
        print(query)
        n = c[1]

        words = preprocessor.preprocess(query)
        q_tf = toTfVector(words, corpusVector)

        k = 0.1
        q_tf_weigth = ((k + 1) * q_tf) / (q_tf + k)
        q_tf_weigth_idf = q_tf_weigth * idf

        sims = []
        for d in documents:
            tf_idf = d[2]
            denom = np.linalg.norm(q_tf_weigth_idf) * np.linalg.norm(tf_idf)

            if denom is 0:
                print("The given query is not present in the collection.")
                continue

            sim = np.dot(q_tf_weigth_idf, tf_idf) / denom
            sims.append((sim, d[1]))

        sims_sorted = sorted(sims, key=lambda tup: -tup[0])
        matrix = evaluate(sims_sorted, c[0], sims_sorted[:n])
        computeScores(matrix)
        print("")


if __name__ == '__main__':
    main()
