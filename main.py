# uncomment if necessary
# nltk.download('stopwords')
# nltk.download('wordnet')

import math
import re
import os

from nltk.corpus import stopwords
from num2words import num2words
from collections import Counter
from tika import parser
from nltk import stem
import numpy as np
import nltk

ps = stem.PorterStemmer()
lm = stem.WordNetLemmatizer()

stops = stopwords.words('english')
letter = re.compile("^[a-z][A-Z]$")
number = re.compile("^[-+]?[0-9]+$")

def preprocess(text):
	# get all the words
	words = re.findall(r'\w+', text.lower())
	
	# remove stop words
	words = [w for w in words if w not in stops]
	
	# remove matched words
	words = [w for w in words if not letter.match(w)]
	
	# convert numbers
	words = [num2words(w) if number.match(w) else w for w in words]
	
	# perform stemming
	words = [ps.stem(w) for w in words]
	
	# perform lemmatization
	words = [lm.lemmatize(w, pos="v") for w in words]
	
	# return a list of words, essentially representing a document
	return words

def toBitVector(d, corpusVector):
	bv = []
	for w in corpusVector:
		bv.append(1 if w in d else 0)
	return np.array(bv)

def logTransform(vector):
	return np.log10(1 + vector)

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
		# vector = logTransform(vector)
		tfs.append(vector)
	return tfs

def createIdfVector(corpusList, tfVectors, documents):
	corpusLen = len(corpusList)
	docsLen = len(documents)
	
	idf = []
	for i in range(corpusLen):
		count = 0
		for tf in tfVectors:
			if tf[i] != 0:
				count = count + 1
		idf.append(np.log10(docsLen / count))
	return np.array(idf)

def createTfIdfVectors(idf, tfVectors):
	tfidfs = []
	for tf in tfVectors:
		tfidfs.append(tf * idf)
	return tfidfs

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
		
			words = preprocess(data)
			
			corpusVector.update(words)
			documents.append(words)
	
	return corpusVector, documents

def main():
	corpusVector, documents = readDataset('./dataset-small')
	
	tfs = createTfVectors(documents, corpusVector)
	idf = createIdfVector(list(corpusVector), tfs, documents)
	tfidfs = createTfIdfVectors(idf, tfs)
		
	print(corpusVector)
	for tfidf in tfidfs:
		print(tfidf)

main()









