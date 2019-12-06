import nltk

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tika import parser

from num2words import num2words
from collections import Counter
import numpy as np
import math
import re
import os

ps = PorterStemmer()
lm = WordNetLemmatizer()

letterPattern = re.compile("^[a-z][A-Z]$")
numberPattern = re.compile("^[-+]?[0-9]+$")
stops = stopwords.words('english')

def bitVector(d, corpusVector):
	bv = []
	for w in corpusVector:
		bv.append(1 if w in d else 0)
	return bv

def tfVector(d, corpusVector):
	tf = []
	counts = Counter(list(d))
	for w in corpusVector:
		tf.append(counts[w])
	return tf

def preprocess(text):
	'''
	Performs pre-processing of the given text and converts it
	into a list of words.
	'''
	
	# get all the words
	words = re.findall(r'\w+', text.lower())
	
	# remove stop words
	words = [w for w in words if w not in stops]
	
	# remove matched words
	words = [w for w in words if not match(w)]
	
	# convert numbers
	words = [num2words(w) if numberPattern.match(w) else w for w in words]
	
	# perform stemming
	words = [ps.stem(w) for w in words]
	
	# perform lemmatization
	words = [lm.lemmatize(w, pos="v") for w in words]
	
	return words

def match(word):
	if letterPattern.match(word):
		return True
	
	return False

def main():
	corpusVector = set()
	documents = []
	counter = 0
	
	for root, subdirs, files in os.walk('./dataset-small'):
		for filename in files:
			filePath = os.path.join(root, filename)
			if not filePath.endswith('.txt'):
				continue

			raw = parser.from_file(filePath)
			data = raw["content"].lower()
		
			words = preprocess(data)
			
			corpusVector.update(words)
			documents.append(words)
	
	tfVectors = []
	for d in documents:
		vector = tfVector(d, corpusVector)
		tfVectors.append(vector)
	
	corpusList = list(corpusVector)
	idf = []
	
	for i in range(len(corpusList)):
		count = 0
		for tf in tfVectors:
			if tf[i] is not 0:
				count = count + 1
		idf.append(math.log(len(documents)/count, 10))
	
	tfidfs = []
	npidf = np.array(idf)
	for tf in tfVectors:
		tfidfs.append(np.array(tf) * npidf)
		
	print(corpusVector)
	for tfidf in tfidfs:
		print(tfidf)

main()









