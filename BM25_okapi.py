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
		#if transform is not None:
		#	vector = transform(vector, k)

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
		createTfVectors(documents, corpusVector, transform=identity)
		tfs = [row[2] for row in documents]

		d_length = [len(row[1]) for row in documents]
		b = 0.5
		k = 0.000001
		okapi = np.array([1-b+b*(np.linalg.norm(row[2])/np.mean(d_length))for row in documents])
		display(okapi)
		print(np.mean(d_length))
		w_tfs = ((k+1)*np.array(tfs))/(np.array(tfs)+k*okapi)       
		idf = createIdfVector(corpusVector, tfs, documents)
		#print('tfs: ',tfs,'     ','w_tfs: ',w_tfs)
		d_length = [len(row[1]) for row in documents]
		#for d in documents:

            
        ###############
		w_idf = createIdfVector(corpusVector, w_tfs, documents)
		for d in documents:
			d[2] = d[2] * w_idf
        ###############
		
		#pickle.dump([tfs, idf, tf_idfs], open(filename, "wb"))
	else:
		#tfs, idf, tf_idfs = pickle.load(open(filename, "rb"))
		pass

	query = "mathematics deals with numbers, patterns, statistics, and game theory astronomy is a branch of science that deals with stars, galaxies, and even physics biology analyzes natural processes, humans, animals, plants, and bacteria"
	
	words = preprocessor.preprocess(query)
	q_tf = toTfVector(words, corpusVector)
	q_tf_idf = q_tf * idf
     
###################################################

	q_tf_weigth =   ((k+1)*q_tf)/(q_tf+k)
	q_tf_weigth_idf = q_tf_weigth * idf
	
	sims = []
	for d in documents:
		w_tf_idf = d[2]
		denom = np.linalg.norm(q_tf_weigth_idf) * np.linalg.norm(w_tf_idf)
		sim = np.dot(q_tf_weigth_idf, w_tf_idf) / denom
		print(sim, d[1])
	print('          ')
		
main()
