from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from TrainModel import *


def text_process(text_raw):
    words = word_tokenize(text_raw)
    words = [word.lower() for word in words if word.lower() not in stopwords.words(language)]

    if word_type == 'surface_no_pm':
        words = [word for word in words if word.isalpha()]

    if word_type == 'stem':
        stemmer = SnowballStemmer(language)
        words = [stemmer.stem(word) for word in words]

    # if word_type == 'suffix':

    return words

