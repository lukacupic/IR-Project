import nltk

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tika import parser
from num2words import num2words
import numpy as np
import re
import os

ps = PorterStemmer()
lm = WordNetLemmatizer()

letterPattern = re.compile("^[a-z][A-Z]$")
numberPattern = re.compile("^[-+]?[0-9]+$")
stops = stopwords.words('english')

def bitVector(d, mainVector):
	bv = []
	for w in mainVector:
		bv.append('1' if w in d else '0')
	return bv

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
	mainVector = set()
	documents = []
	counter = 0
	
	for root, subdirs, files in os.walk('./dataset'):
		if counter is 5:
			break
			
		for filename in files:
			if counter is 5:
				break
			
			filePath = os.path.join(root, filename)

			raw = parser.from_file(filePath)
			data = raw["content"].lower()
		
			words = preprocess(data)
			
			mainVector.update(words)
			documents.append(words)
			
			counter = counter + 1
	
	mainList = list(mainVector)
	
	#for d in documents:
		#vector = bitVector(d, mainVector)
		
	print(bitVector(documents[0], mainVector))

main()









