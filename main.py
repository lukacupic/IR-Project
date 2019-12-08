import importlib
import math
import os

from collections import Counter
from tika import parser
import numpy as np

from preprocess import Preprocessor

def toBitVector(d, corpusVector):
	bv = []
	for w in corpusVector:
		bv.append(1 if w in d else 0)
	return np.array(bv)

def logTransform(vector):
	return np.log10(1 + vector)

def bm25Transform(vector, k):
	return ((k + 1) * vector) / (k + vector)

def toTfVector(d, corpusVector):
	tf = []
	counts = Counter(d)
	for w in corpusVector:
		tf.append(counts[w])
	return np.array(tf)

def createTfVectors(documents, corpusVector):
	tfs = []
	for d in documents:
		vector = toTfVector(d, corpusVector)
		#vector = logTransform(vector)
		#vector = bm25Transform(vector, k=0.85)
		tfs.append(vector)
	return tfs

def createIdfVector(corpusVector, tfs, documents):
	corpusLen = len(corpusVector)
	docsLen = len(documents)
	
	idf = []
	for i in range(corpusLen):
		count = 0
		for tf in tfs:
			if tf[i] != 0:
				count = count + 1
		idf.append(np.log10(docsLen / count))
	return np.array(idf)

def createTfIdfVectors(tfs, idf):
	tfidfs = []
	for tf in tfs:
		tfidfs.append(tf * idf)
	return tfidfs

def readDataset(path):
	corpusVector = set()
	documents = []
	counter = 0
	
	preprocessor = Preprocessor()
	
	for root, subdirs, files in os.walk(path):
		for filename in files:
			filePath = os.path.join(root, filename)
			
			if not filePath.endswith('.txt'):
				continue

			raw = parser.from_file(filePath)
			data = raw["content"].lower()
		
			words = preprocessor.preprocess(data)
			
			corpusVector.update(words)
			documents.append(words)
	
	return corpusVector, documents

def main():
	corpusVector, documents = readDataset('./dataset-small')
	
	tfs = createTfVectors(documents, corpusVector)
	idf = createIdfVector(corpusVector, tfs, documents)
	tfidfs = createTfIdfVectors(tfs, idf)
		
	print(corpusVector)
	for tfidf in tfidfs:
		print(tfidf)

main()









