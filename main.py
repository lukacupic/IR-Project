import os

from tika import parser
import pickle

from document import Document
from preprocess import Preprocessor
from transform import *
from method import *


def createVectors(documents, method):
    for d in documents:
        tf = method.getVector(d.getWords())
        d.setTf(tf)
    return


def setVectors(documents, tfs, idf):
    for i in range(len(documents)):
        documents[i].setTf(tfs[i])
        documents[i].setTfIdf(tfs[i] * idf)
    return


def readDataset(path, preprocessor):
    corpusVector = []
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

            print("Reading file %d..." % file)
            file = file + 1

            if not filePath.endswith('.pdf') and not filePath.endswith('.txt'):
                continue

            raw = parser.from_file(filePath)
            data = raw["content"].lower()

            words = preprocessor.preprocess(data)

            for w in words:
                if w not in corpusVector:
                    corpusVector.append(w)

            d = Document(words, filePath, None, None)
            documents.append(d)

    return corpusVector, documents, categories


# does the document exist in the top N documents?
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
    fn = matrix[1][0]
    tn = matrix[1][1]

    acc = (tp + tn) / np.float64(tp + tn + fp + fn)
    pre = tp / np.float64(tp + fp)
    rec = tp / np.float64(tp + fn)
    f1 = 2 * tp / np.float64(2 * tp + fp + fn)

    print("Accuracy:  %.4f" % acc)
    print("Precision: %.4f" % pre)
    print("Recall:    %.4f" % rec)
    print("F1:        %.4f" % f1)
    return


def main():
    preprocessor = Preprocessor()

    # transform = IdentityTransform()
    # transform = BM25Transform(k=1.4)
    # transform = BM25OkapiTransform(k=1.4, b=0.75)
    transform = LogTransform()

    # method = BitVector()
    # method = Tf()
    method = TfIdf()

    datasetName = "dataset"
    vectorsFile = "./pickle/" + datasetName + "-" + method.getName() + "-" + transform.getName() + ".pickle"
    documentsFile = "./pickle/" + datasetName + "-documents.pickle"

    # check for documents
    if not os.path.exists(documentsFile):
        corpusVector, documents, categories = readDataset("./" + datasetName, preprocessor)
        pickle.dump([corpusVector, documents, categories], open(documentsFile, "wb"))
    else:
        corpusVector, documents, categories = pickle.load(open(documentsFile, "rb"))

    method.setCorpusVector(corpusVector)
    transform.setDocuments(documents)

    # check for vectors
    if not os.path.exists(vectorsFile):
        createVectors(documents, method)
        transform.transformDocuments(documents)

        tfs = [d.getTf() for d in documents]
        idf = method.getIdf(tfs, documents)

        for d in documents:
            d.setTfIdf(d.getTf() * idf)

        pickle.dump([tfs, idf], open(vectorsFile, "wb"))

    else:
        tfs, idf = pickle.load(open(vectorsFile, "rb"))
        setVectors(documents, tfs, idf)

    for c in categories:
        query = os.path.basename(c[0])
        print(query)
        n = c[1]

        words = preprocessor.preprocess(query)

        tf_q = transform.transform(method.getVector(words))
        tf_idf_q = tf_q * idf

        sims = []
        for d in documents:
            tf_idf = d.getTfIdf()
            denom = np.linalg.norm(tf_idf_q) * np.linalg.norm(tf_idf)

            if denom is 0:
                print("The given query is not present in the collection.")
                continue

            sim = np.dot(tf_idf_q, tf_idf) / denom
            sims.append((sim, d.getPath()))

        sum = 0
        for d in documents:
            sum = sum + np.sum(d.getTfIdf())

        sims_sorted = sorted(sims, key=lambda tup: -tup[0])
        matrix = evaluate(sims_sorted, c[0], sims_sorted[:n])
        computeScores(matrix)
        print("")


if __name__ == '__main__':
    main()
