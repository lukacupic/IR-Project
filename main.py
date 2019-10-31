import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tika import parser
import numpy as np
import re

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
	
	stemmed = [ps.stem(w) for w in filtered]
	return stemmed

def main():
	raw = parser.from_file('dataset/astro.pdf')
	data = raw["content"].lower()
	print(preprocess(data))

main()
