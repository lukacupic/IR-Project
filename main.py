import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tika import parser
import numpy as np
import re

numberPattern = re.compile("^[-+]?[0-9]+$")
singlePattern = re.compile("^[a-z][A-Z][0-9]$")

ps = PorterStemmer()

def preprocess(text):
	'''
	Performs pre-processing of the given text and converts it 
	into a list of words.
	'''
	
	text = text.lower()
	words = re.findall(r'\w+', text)
	
	stops = stopwords.words('english')
	filtered = [w for w in words if w not in stops]
	filtered = [w for w in filtered if not match(w)]
	
	stemmed = [ps.stem(w) for w in filtered]
	return stemmed

def match(word):
	if numberPattern.match(word):
		return True

	if singlePattern.match(word):
		return True

def main():
	raw = parser.from_file('dataset/astro.pdf')
	data = raw["content"].lower()
	
	words = preprocess(data)
	print(words)

main()
