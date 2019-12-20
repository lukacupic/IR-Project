import os

from tika import parser
import pickle

from document import Document
from preprocess import Preprocessor
from transform import *
from method import *

preprocessor = Preprocessor()


def createVectors(documents, method):
    for d in documents:
        tf = method.getVector(d.getWords())
        d.setTf(tf)


def readDataset(path):
    corpusVector = set()
    documents = []
    counter = 0
    categories = []

    file = 1

    for root, subdirs, files in os.walk(path):
        if counter != 0:
            categories.append([root, len(files)])

        counter = counter + 1
        for filename in files:
            filePath = os.path.join(root, filename)

            print("Reading file %d/148" % file)
            file = file + 1

            if not filePath.endswith('.pdf') and not filePath.endswith('.txt'):
                continue

            raw = parser.from_file(filePath)
            data = raw["content"].lower()

            words = preprocessor.preprocess(data)

            corpusVector.update(words)
            d = Document(words, filePath, None, None)
            documents.append(d)

    return corpusVector, documents, categories


def isRelevant(document, top_n):
    for d in top_n:
        if document[1] == d[1]:
            return True
    return False


def evaluate(docs, path, top_n):
    matrix = [[0, 0], [0, 0]]

    for d in docs:
        relevant = isRelevant(d, top_n)
        dPath = d[1]

        if path in dPath and relevant:
            matrix[0][0] = matrix[0][0] + 1

        if path not in dPath and relevant:
            matrix[0][1] = matrix[0][1] + 1

        if path in dPath and not relevant:
            matrix[1][0] = matrix[1][0] + 1

        if path not in dPath and not relevant:
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
    transform = IdentityTransform()
    # transform = BM25Transform(k=0.1)
    # transform = LogTransform()

    # method = BitVector()
    # method = Tf()
    method = TfIdf()

    filename = method.getName() + "-" + transform.getName() + ".pickle"

    if not os.path.exists(filename):
        corpusVector, documents, categories = readDataset('./dataset')
        # pickle.dump([corpusVector, documents, categories], open("./bla", "wb"))
        # corpusVector, documents, categories = pickle.load(open("./bla", "rb"))

        method.setCorpusVector(corpusVector)

        createVectors(documents, method)
        transform.transformDocuments(documents)

        tfs = [d.getTf() for d in documents]
        idf = method.getIdf(tfs, documents)

        for d in documents:
            d.setTfIdf(d.getTf() * idf)

        pickle.dump([corpusVector, documents, categories, tfs, idf], open(filename, "wb"))

    else:
        corpusVector, documents, categories, tfs, idf = pickle.load(open(filename, "rb"))
        method.setCorpusVector(corpusVector)

    for c in categories:
        query = os.path.basename(c[0])
        print(query)
        n = c[1]

        words = preprocessor.preprocess(query)
        q_tf = method.getVector(words)

        q_tf_weigth = transform.transform(q_tf)
        q_tf_weigth_idf = q_tf_weigth * idf

        sims = []
        for d in documents:
            tf_idf = d.getTfIdf()
            denom = np.linalg.norm(q_tf_weigth_idf) * np.linalg.norm(tf_idf)

            if denom is 0:
                print("The given query is not present in the collection.")
                continue

            sim = np.dot(q_tf_weigth_idf, tf_idf) / denom
            sims.append((sim, d.getPath()))

        sims_sorted = sorted(sims, key=lambda tup: -tup[0])
        matrix = evaluate(sims_sorted, c[0], sims_sorted[:n])
        computeScores(matrix)
        print("")


if __name__ == '__main__':
    main()
