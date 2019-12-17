import importlib
import math
import os

from collections import Counter
from tika import parser
import numpy as np
import pickle

from preprocess import Preprocessor

#-------------------------------------

# TODO clean up the variables here
filename = "data-tf_idf.pickle"

preprocessor = Preprocessor()

#-------------------------------------

def toBitVector(d, corpusVector):
	bv = []
	for w in corpusVector:
		bv.append(1 if w in d else 0)
	return np.array(bv)

def identity(vector, k):
	return vector

def logTransform(vector, k):
	return np.log10(1 + vector)

def bm25Transform(vector, k):
	return ((k + 1) * vector) / (k + vector)

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
		idf.append(np.log10(docsLen / count))
	return np.array(idf)

def readDataset(path):
	corpusVector = set()
	documents = []
	counter = 0
	
	for root, subdirs, files in os.walk(path):
		for filename in files:
			filePath = os.path.join(root, filename)
			
			if not filePath.endswith('.txt'):
				continue

			raw = parser.from_file(filePath)
			data = raw["content"].lower()
		
			words = preprocessor.preprocess(data)
			
			corpusVector.update(words)
			documents.append([words, filePath])
	
	return corpusVector, documents

def main():
	corpusVector, documents = readDataset('./dataset-small')
	
	if not os.path.exists(filename):
		createTfVectors(documents, corpusVector, transform=bm25Transform, k=0.1)
		tfs = [row[2] for row in documents]
		idf = createIdfVector(corpusVector, tfs, documents)
		
		for d in documents:
			d[2] = d[2] * idf
		
		#pickle.dump([tfs, idf, tf_idfs], open(filename, "wb"))
	else:
		#tfs, idf, tf_idfs = pickle.load(open(filename, "rb"))
		pass
	
	xyz = [row[2] for row in documents]
	print(np.sum(xyz))

	query = "big dragons"
	
	words = preprocessor.preprocess(query)
	q_tf = toTfVector(words, corpusVector)
	q_tf_idf = q_tf * idf
	
	k = 0.1
	q_tf_weigth =   ((k+1)*q_tf)/(q_tf+k)
	q_tf_weigth_idf = q_tf_weigth * idf
	q_tf_idf = q_tf_weigth_idf
	
	for d in documents:
		tf_idf = d[2]
		denom = np.linalg.norm(q_tf_idf) * np.linalg.norm(tf_idf)
		sim = np.dot(q_tf_idf, tf_idf) / denom
		print(sim, d[1])
		
def main_bm25():
	corpusVector, documents = readDataset('./dataset-small')

	if not os.path.exists(filename):
		createTfVectors(documents, corpusVector, transform=identity)
		tfs = [row[2] for row in documents]
		
		k = 0.1
		w_tfs = ((k+1)*np.array(tfs))/(np.array(tfs)+k)
		idf = createIdfVector(corpusVector, tfs, documents)
		
		w_idf = createIdfVector(corpusVector, w_tfs, documents)
		for d in documents:
			d[2] = d[2] * w_idf
		
		#pickle.dump([tfs, idf, tf_idfs], open(filename, "wb"))
	else:
		#tfs, idf, tf_idfs = pickle.load(open(filename, "rb"))
		pass
	

	xyz = [row[2] for row in documents]
	print(np.sum(xyz))
	
	query = "big dragons"
	
	words = preprocessor.preprocess(query)
	q_tf = toTfVector(words, corpusVector)

	q_tf_weigth =   ((k+1)*q_tf)/(q_tf+k)
	q_tf_weigth_idf = q_tf_weigth * idf
	#print('tfs: ',q_tf,'     ','w_tfs: ',q_tf_weigth)
	
	for d in documents:
		w_tf_idf = d[2]
		denom = np.linalg.norm(q_tf_weigth_idf) * np.linalg.norm(w_tf_idf)
		sim = np.dot(q_tf_weigth_idf, w_tf_idf) / denom
		print(sim, d[1])

#main()
main_bm25()
#main_okapi()
