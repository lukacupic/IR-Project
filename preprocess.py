# uncomment if necessary
# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.corpus import stopwords
from num2words import num2words
from nltk import stem
import nltk
import re

class Preprocessor:
	
	stopWords = "stop_words"
	letters = "letters"
	numbers = "numbers"
	stemming = "stemming"
	lemming = "lemming"
	
	stops = stopwords.words('english')
	letters = re.compile("^[a-z][A-Z]$")
	number = re.compile("^[-+]?[0-9]+$")
	
	ps = stem.PorterStemmer()
	lm = stem.WordNetLemmatizer()
	
	def __init__(self, methods=[stopWords, letters, numbers, lemming]):
		self.methods = methods
		
	def preprocess(self, text):
		words = re.findall(r'\w+', text.lower())
		
		if self.stopWords in self.methods:
			words = [w for w in words if w not in self.stops]
		
		if self.letters in self.methods:
			words = [w for w in words if not self.letters.match(w)]
		
		if self.numbers in self.methods:
			words = [num2words(w) if self.number.match(w) else w for w in words]
		
		if self.stemming in self.methods:
			words = [self.ps.stem(w) for w in words]
		
		if self.lemming in self.methods:
			words = [self.lm.lemmatize(w, pos="v") for w in words]
		
		return words
